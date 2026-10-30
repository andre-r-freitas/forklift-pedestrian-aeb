#include "YOLOInferenceActor.h"
#include "Engine/TextureRenderTarget2D.h"
#include "EngineUtils.h"
#include "RenderUtils.h"
#include "RenderingThread.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNEModelData.h"
#include "Serialization/BufferArchive.h"
#include "ImageUtils.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "Camera/CameraComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/Engine.h"
#include "Engine/EngineTypes.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/DateTime.h"
#include "Utils/GroundTruthUtils.h"
#include "Misc/Crc.h"

AYOLOInferenceActor::AYOLOInferenceActor()
{
    PrimaryActorTick.bCanEverTick = false;
}

void AYOLOInferenceActor::BeginPlay()
{
    Super::BeginPlay();
    UE_LOG(LogTemp, Log, TEXT("AYOLOInferenceActor BeginPlay called, loading model..."));
    LoadModel();

    if (CopyTextureMaterial)
    {
        CopyTextureMID = UMaterialInstanceDynamic::Create(CopyTextureMaterial, this);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("CopyTextureMaterial is not assigned in Blueprint."));
    }

    GroundTruthAssignments = GroundTruthUtils::AssignStencilIDsToTaggedActors(GetWorld(), GroundTruthActorTag, GroundTruthClassID);

    for (const TPair<int32, int32>& Pair : GroundTruthAssignments.StencilIDToClassID)
    {
        UE_LOG(LogTemp, Log, TEXT("Assigned map: Stencil %d -> Class %d"), Pair.Key, Pair.Value);
    }

    // No output folder is created here any more: runs are started by BeginEvaluationRun
    // (ScenarioManager), or lazily by EnsureRunActive() when inference is triggered without one.
    FrameCounter = 0;
}

void AYOLOInferenceActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (Logger.IsActive())
    {
        EndEvaluationRun(RunMetadataProvider ? RunMetadataProvider() : FString(TEXT("\"end_reason\": \"EndPlay\"")));
    }

    Super::EndPlay(EndPlayReason);
}

double AYOLOInferenceActor::NowSeconds() const
{
    return GetWorld() ? GetWorld()->GetTimeSeconds() : 0.0;
}

// ============================================================================================
// Run management
// ============================================================================================

void AYOLOInferenceActor::BeginEvaluationRun(const FString& BatchId, const FString& ScenarioId, int32 Repetition, AActor* InForklift)
{
    if (Logger.IsActive())
    {
        EndEvaluationRun(TEXT("\"end_reason\": \"Superseded\""));
    }

    ++RunSerial;            // results from an older run arriving late (async mode) are dropped
    FrameCounter = 0;
    bInferenceRunning = false;

    if (InForklift)
    {
        ForkliftActor = InForklift;
    }

    CaptureComponent = FindCaptureComponent();
    if (!CaptureComponent.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("YOLO: no SceneCaptureComponent2D writing to InputRenderTarget found on the forklift or this actor. Camera pose will not be logged."));
    }

    SafetyController = ForkliftActor ? ForkliftActor->FindComponentByClass<USafetyBrakeController>() : nullptr;

    if (!DepthRenderTarget)
    {
        UE_LOG(LogTemp, Warning, TEXT("YOLO: no DepthRenderTarget set: distances are not estimated and the safety controller receives no observations."));
    }

    ForcedCaptureComponents.Reset();   // re-collected on the first frame of the run
    StaleFrameCount = 0;
    bHasLastFrameCrc = false;

    const FString RunId = FString::Printf(TEXT("%s_rep%02d"), *ScenarioId, Repetition);
    Logger.BeginRun(BatchId, RunId, ScenarioId, Repetition);
    RunStartTimeSeconds = NowSeconds();

    UE_LOG(LogTemp, Log, TEXT("YOLO: evaluation run started: %s"), *Logger.GetRunFolder());
}

void AYOLOInferenceActor::EndEvaluationRun(const FString& ExtraMetadataJsonFields)
{
    if (!Logger.IsActive())
    {
        return;
    }

    const int32 ImageWidth = DisplayRenderTarget ? DisplayRenderTarget->SizeX : 0;
    const int32 ImageHeight = DisplayRenderTarget ? DisplayRenderTarget->SizeY : 0;

    FString Json = FString::Printf(
        TEXT("{\n")
        TEXT("  \"run_id\": \"%s\",\n")
        TEXT("  \"scenario_id\": \"%s\",\n")
        TEXT("  \"repetition\": %d,\n")
        TEXT("  \"frames_logged\": %d,\n")
        TEXT("  \"run_start_s\": %.4f,\n")
        TEXT("  \"run_end_s\": %.4f,\n")
        TEXT("  \"image_width\": %d,\n")
        TEXT("  \"image_height\": %d,\n")
        TEXT("  \"model_asset\": \"%s\",\n")
        TEXT("  \"confidence_threshold\": %.3f,\n")
        TEXT("  \"nms_iou_threshold\": %.3f,\n")
        TEXT("  \"match_iou_threshold\": %.3f,\n")
        TEXT("  \"person_class_id\": %d,\n")
        TEXT("  \"synchronous_inference\": %s,\n")
        TEXT("  \"full_silhouette_from_g\": %s,\n")
        TEXT("  \"depth_enabled\": %s,\n")
        TEXT("  \"depth_percentile\": %.2f,\n")
        TEXT("  \"min_valid_depth_m\": %.2f,\n")
        TEXT("  \"camera_component\": \"%s\",\n")
        TEXT("  \"forced_scene_capture\": %s,\n")
        TEXT("  \"stale_frames\": %d,\n")
        TEXT("  \"clean_frames_saved\": %s,\n")
        TEXT("  \"depth_frames_saved\": %s,\n")
        TEXT("  \"ground_truth_frames_saved\": %s,\n")
        TEXT("  \"depth_png\": \"uint16 mm, planar depth along camera axis, 0 = invalid\""),
        *Logger.GetRunId(),
        *Logger.GetScenarioId(),
        Logger.GetRepetition(),
        Logger.GetFramesLogged(),
        RunStartTimeSeconds,
        NowSeconds(),
        ImageWidth,
        ImageHeight,
        *ModelAssetPath,
        ConfidenceThreshold,
        NMSIoUThreshold,
        MatchIoUThreshold,
        PersonClassID,
        bSynchronousInference ? TEXT("true") : TEXT("false"),
        bReadFullSilhouetteFromGreenChannel ? TEXT("true") : TEXT("false"),
        DepthRenderTarget ? TEXT("true") : TEXT("false"),
        DepthPercentile,
        MinValidDepthM,
        CaptureComponent.IsValid() ? *CaptureComponent->GetName() : TEXT(""),
        bForceSceneCaptureBeforeInference ? TEXT("true") : TEXT("false"),
        StaleFrameCount,
        bSaveCleanFrames ? TEXT("true") : TEXT("false"),
        (bSaveDepthFrames && DepthRenderTarget) ? TEXT("true") : TEXT("false"),
        (bSaveGroundTruthFrames && GroundTruthRenderTarget) ? TEXT("true") : TEXT("false"));

    if (!ExtraMetadataJsonFields.IsEmpty())
    {
        Json += TEXT(",\n  ") + ExtraMetadataJsonFields;
    }
    Json += TEXT("\n}\n");

    Logger.EndRun(Json);
    ++RunSerial; // drop any async result still in flight
    bInferenceRunning = false;
    RunMetadataProvider = nullptr;
}

void AYOLOInferenceActor::EnsureRunActive()
{
    if (Logger.IsActive())
    {
        return;
    }

    const FString BatchId = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
    BeginEvaluationRun(BatchId, TEXT("Standalone"), 0, ForkliftActor);
}

void AYOLOInferenceActor::ForceSceneCaptures()
{
    UWorld* World = GetWorld();
    if (!World)
    {
        return;
    }

    if (ForcedCaptureComponents.Num() == 0)
    {
        // Every capture component in the level that writes one of the render targets we read.
        bool bFoundInput = false, bFoundGT = false, bFoundDepth = false;
        for (TActorIterator<AActor> It(World); It; ++It)
        {
            TArray<USceneCaptureComponent2D*> Captures;
            It->GetComponents<USceneCaptureComponent2D>(Captures);
            for (USceneCaptureComponent2D* Capture : Captures)
            {
                if (!Capture || !Capture->TextureTarget)
                {
                    continue;
                }
                const bool bInput = Capture->TextureTarget == InputRenderTarget;
                const bool bGT = GroundTruthRenderTarget && Capture->TextureTarget == GroundTruthRenderTarget;
                const bool bDepth = DepthRenderTarget && Capture->TextureTarget == DepthRenderTarget;
                if (bInput || bGT || bDepth)
                {
                    ForcedCaptureComponents.Add(Capture);
                    bFoundInput |= bInput;
                    bFoundGT |= bGT;
                    bFoundDepth |= bDepth;
                }
            }
        }

        if (!bFoundInput || (GroundTruthRenderTarget && !bFoundGT) || (DepthRenderTarget && !bFoundDepth))
        {
            UE_LOG(LogTemp, Warning, TEXT("YOLO: forced capture: no capture component found for%s%s%s. Those images may be stale."),
                bFoundInput ? TEXT("") : TEXT(" RGB"),
                (GroundTruthRenderTarget && !bFoundGT) ? TEXT(" ground-truth") : TEXT(""),
                (DepthRenderTarget && !bFoundDepth) ? TEXT(" depth") : TEXT(""));
        }
    }

    for (const TWeakObjectPtr<USceneCaptureComponent2D>& Capture : ForcedCaptureComponents)
    {
        if (Capture.IsValid())
        {
            // Enqueued on the render thread before the copy/readback commands below, so they see this frame.
            Capture->CaptureScene();
        }
    }
}

USceneComponent* AYOLOInferenceActor::FindCaptureComponent() const
{
    TArray<const AActor*> Candidates;
    if (ForkliftActor)
    {
        Candidates.Add(ForkliftActor.Get());
    }
    Candidates.Add(this);

    USceneCaptureComponent2D* Fallback = nullptr;

    for (const AActor* Candidate : Candidates)
    {
        TArray<USceneCaptureComponent2D*> Captures;
        Candidate->GetComponents<USceneCaptureComponent2D>(Captures);

        for (USceneCaptureComponent2D* Capture : Captures)
        {
            if (Capture && Capture->TextureTarget == InputRenderTarget)
            {
                return Capture;
            }
            if (!Fallback)
            {
                Fallback = Capture;
            }
        }
    }

    return Fallback;
}

// ============================================================================================
// World state (ground truth for distance, velocity and time-to-collision)
// ============================================================================================

FString AYOLOInferenceActor::CollectWorldStateRows(int32 FrameIndex, double TimestampSeconds) const
{
    const FString Prefix = Logger.RowPrefix(FrameIndex, TimestampSeconds);
    FString Rows;

    auto AddRow = [&Rows, &Prefix](const TCHAR* RoleName, const FString& Name, int32 StencilID,
        const FVector& LocationCm, const FVector& VelocityCmS, const FRotator& Rotation)
        {
            Rows += Prefix + FString::Printf(
                TEXT("%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f\n"),
                RoleName, *Name, StencilID,
                LocationCm.X / 100.0, LocationCm.Y / 100.0, LocationCm.Z / 100.0,
                VelocityCmS.X / 100.0, VelocityCmS.Y / 100.0, VelocityCmS.Z / 100.0,
                Rotation.Yaw, Rotation.Pitch);
        };

    FVector ForkliftVelocity = FVector::ZeroVector;

    if (ForkliftActor)
    {
        ForkliftVelocity = ForkliftActor->GetVelocity();
        AddRow(TEXT("forklift"), ForkliftActor->GetName(), 0,
            ForkliftActor->GetActorLocation(), ForkliftVelocity, ForkliftActor->GetActorRotation());
    }

    if (CaptureComponent.IsValid())
    {
        // The camera is rigidly mounted on the forklift, so it moves with the forklift's velocity.
        AddRow(TEXT("camera"), CaptureComponent->GetName(), 0,
            CaptureComponent->GetComponentLocation(), ForkliftVelocity, CaptureComponent->GetComponentRotation());
    }

    for (const TPair<int32, TWeakObjectPtr<AActor>>& Pair : GroundTruthAssignments.StencilIDToActor)
    {
        const AActor* Pedestrian = Pair.Value.Get();
        if (!Pedestrian)
        {
            continue;
        }

        AddRow(TEXT("pedestrian"), Pedestrian->GetName(), Pair.Key,
            Pedestrian->GetActorLocation(), Pedestrian->GetVelocity(), Pedestrian->GetActorRotation());
    }

    return Rows;
}

// ============================================================================================
// Model + pre/post-processing
// ============================================================================================

void AYOLOInferenceActor::SaveTensorToRaw(const TArray<float>& Tensor, const FString& FilePath)
{
    FBufferArchive Archive;
    Archive.Append((const uint8*)Tensor.GetData(), Tensor.Num() * sizeof(float));
    FFileHelper::SaveArrayToFile(Archive, *FilePath);
    Archive.FlushCache();
    Archive.Empty();
}

bool AYOLOInferenceActor::LoadModel()
{
    if (!ModelData)
    {
        ModelData = LoadObject<UNNEModelData>(nullptr, *ModelAssetPath);

        if (!ModelData)
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to load YOLO model at path: %s"), *ModelAssetPath);
            return false;
        }
    }
    TArray<FString> AvailableGPURuntimes = UE::NNE::GetAllRuntimeNames<INNERuntimeGPU>();
    for (const FString& RuntimeName : AvailableGPURuntimes)
    {
        UE_LOG(LogTemp, Log, TEXT("Available GPU runtime: %s"), *RuntimeName);
    }
    FString RuntimeNames[] = {
        TEXT("NNERuntimeORTDml"),
        TEXT("NNERuntimeGPU"),
        TEXT("NNERuntimeORTDirectML")
    };
    TWeakInterfacePtr<INNERuntimeGPU> Runtime;
    for (const FString& Name : RuntimeNames) {
        Runtime = UE::NNE::GetRuntime<INNERuntimeGPU>(Name);
        if (Runtime.IsValid()) {
            UE_LOG(LogTemp, Log, TEXT("Runtime: %s"), *Name);
            break;
        }
    }
    if (!Runtime.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to get NNE runtime"));
        return false;
    }
    TSharedPtr<UE::NNE::IModelGPU> Model = Runtime->CreateModelGPU(ModelData);
    if (!Model.IsValid()) return false;
    ModelInstance = Model->CreateModelInstanceGPU();
    return ModelInstance.IsValid();
}

UTexture2D* AYOLOInferenceActor::CreateTextureFromPixels(const TArray<FColor>& Pixels, int32 Width, int32 Height)
{
    if (Pixels.Num() != Width * Height)
    {
        UE_LOG(LogTemp, Error, TEXT("Pixel count mismatch: %d vs expected %d"), Pixels.Num(), Width * Height);
        return nullptr;
    }

    UTexture2D* Texture = UTexture2D::CreateTransient(Width, Height, PF_B8G8R8A8);
    if (!Texture)
    {
        UE_LOG(LogTemp, Error, TEXT("CreateTransient failed"));
        return nullptr;
    }
    Texture->SRGB = true;
    Texture->CompressionSettings = TC_VectorDisplacementmap;
    Texture->MipGenSettings = TMGS_NoMipmaps;
    Texture->UpdateResource();

    const int32 TexWidth = Texture->GetSizeX();
    const int32 TexHeight = Texture->GetSizeY();

    if (TexWidth != Width || TexHeight != Height)
    {
        UE_LOG(LogTemp, Error, TEXT("Texture created with wrong size! %dx%d vs input %dx%d"), TexWidth, TexHeight, Width, Height);
        return nullptr;
    }

    FUpdateTextureRegion2D Region(0, 0, 0, 0, TexWidth, TexHeight);

    Texture->UpdateTextureRegions(
        0,
        1,
        &Region,
        TexWidth * sizeof(FColor),
        sizeof(FColor),
        reinterpret_cast<uint8*>(const_cast<FColor*>(Pixels.GetData())),
        nullptr
    );
    return Texture;
}

bool AYOLOInferenceActor::PreprocessRenderTarget(UTextureRenderTarget2D* RenderTarget, TArray<float>& TensorData, TArray<FColor>* OutPixels)
{
    FTextureRenderTargetResource* RTResource = RenderTarget->GameThread_GetRenderTargetResource();
    if (!RTResource) return false;

    TArray<FColor> Pixels;
    RTResource->ReadPixels(Pixels);

    const int32 Height = RenderTarget->SizeY;
    const int32 Width = RenderTarget->SizeX;

    if (Pixels.Num() != Width * Height)
    {
        UE_LOG(LogTemp, Error, TEXT("Pixel count mismatch after ReadPixels"));
        return false;
    }

    if (OutPixels)
    {
        *OutPixels = Pixels;
    }

    TensorData.SetNum(3 * Height * Width); // NCHW, channel-first

    // FIX: Ultralytics ONNX models expect RGB input (their Python preprocessing converts
    // OpenCV's BGR to RGB before inference). Channel 0 = R, 1 = G, 2 = B, values in [0, 1].
    const int32 PlaneSize = Height * Width;
    for (int32 PixelIndex = 0; PixelIndex < PlaneSize; ++PixelIndex)
    {
        const FColor& Pixel = Pixels[PixelIndex];
        TensorData[0 * PlaneSize + PixelIndex] = Pixel.R / 255.0f;
        TensorData[1 * PlaneSize + PixelIndex] = Pixel.G / 255.0f;
        TensorData[2 * PlaneSize + PixelIndex] = Pixel.B / 255.0f;
    }

    return true;
}

TArray<FBoundingBox2> AYOLOInferenceActor::ParseYOLOOutput(const TArray<float>& OutputData)
{
    TArray<FBoundingBox2> Boxes;
    constexpr int32 Channels = 84;        // cx, cy, w, h + 80 COCO class scores
    constexpr int32 TotalAnchors = 8400;  // 640x640 input

    if (OutputData.Num() < Channels * TotalAnchors)
    {
        UE_LOG(LogTemp, Error, TEXT("YOLO: output too small (%d values)"), OutputData.Num());
        return Boxes;
    }

    for (int32 AnchorIdx = 0; AnchorIdx < TotalAnchors; AnchorIdx++)
    {
        const float Cx = OutputData[AnchorIdx + 0 * TotalAnchors];
        const float Cy = OutputData[AnchorIdx + 1 * TotalAnchors];
        const float W = OutputData[AnchorIdx + 2 * TotalAnchors];
        const float H = OutputData[AnchorIdx + 3 * TotalAnchors];

        int32 BestClass = -1;
        float BestClassConf = 0.0f;

        for (int32 ClassIdx = 4; ClassIdx < Channels; ClassIdx++)
        {
            const float ClassConf = OutputData[AnchorIdx + ClassIdx * TotalAnchors];
            if (ClassConf > BestClassConf)
            {
                BestClassConf = ClassConf;
                BestClass = ClassIdx - 4;
            }
        }

        if (BestClassConf > ConfidenceThreshold)
        {
            FBoundingBox2 Box;
            Box.CenterX = Cx;
            Box.CenterY = Cy;
            Box.Width = W;
            Box.Height = H;
            Box.Confidence = BestClassConf;
            Box.ClassID = BestClass;
            Boxes.Add(Box);
        }
    }

    return Boxes;
}

bool AYOLOInferenceActor::CopyToDisplayRenderTarget()
{
    if (!CopyTextureMID)
    {
        UE_LOG(LogTemp, Warning, TEXT("CopyTextureMID is null!"));
        return false;
    }
    if (!DisplayRenderTarget)
    {
        UE_LOG(LogTemp, Warning, TEXT("DisplayRenderTarget is null!"));
        return false;
    }
    if (!InputRenderTarget)
    {
        UE_LOG(LogTemp, Warning, TEXT("InputRenderTarget is null!"));
        return false;
    }

    CopyTextureMID->SetTextureParameterValue(TEXT("InputTexture"), InputRenderTarget);
    UKismetRenderingLibrary::DrawMaterialToRenderTarget(GetWorld(), DisplayRenderTarget, CopyTextureMID);

    return true;
}

// ============================================================================================
// Inference
// ============================================================================================

void AYOLOInferenceActor::RunYOLOInference()
{
    if (bInferenceRunning)
    {
        // Only possible in asynchronous mode. The frame is skipped and NOT counted.
        UE_LOG(LogTemp, Verbose, TEXT("Inference already running, skipping this request"));
        return;
    }

    if (!ModelInstance.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("YOLO: model instance is not valid (did LoadModel fail?)."));
        return;
    }

    if (InputRenderTarget == nullptr || DisplayRenderTarget == nullptr)
    {
        UE_LOG(LogTemp, Warning, TEXT("YOLO: InputRenderTarget or DisplayRenderTarget not set!"));
        return;
    }

    EnsureRunActive();

    if (bForceSceneCaptureBeforeInference)
    {
        ForceSceneCaptures();
    }

    if (!CopyToDisplayRenderTarget())
    {
        return;
    }

    // ---- Capture everything at the same moment ----
    TSharedPtr<FYOLOCaptureSnapshot> Snapshot = MakeShared<FYOLOCaptureSnapshot>();

    TArray<float> InputTensorData;
    if (!PreprocessRenderTarget(DisplayRenderTarget, InputTensorData, &Snapshot->Pixels))
    {
        return;
    }

    // Stale-frame check: an identical image while the forklift moves means the render target was not updated.
    {
        const uint32 Crc = FCrc::MemCrc32(Snapshot->Pixels.GetData(), Snapshot->Pixels.Num() * sizeof(FColor));
        const float ForkliftSpeedMps = ForkliftActor ? static_cast<float>(ForkliftActor->GetVelocity().Size() / 100.0) : 0.0f;
        Snapshot->bStaleFrame = bHasLastFrameCrc && Crc == LastFrameCrc && ForkliftSpeedMps > 0.2f;
        if (Snapshot->bStaleFrame)
        {
            if (StaleFrameCount == 0)
            {
                UE_LOG(LogTemp, Warning, TEXT("YOLO: STALE FRAME - the camera image did not change while the forklift moves (%.2f m/s). Is the editor/PIE window in the background?"), ForkliftSpeedMps);
            }
            ++StaleFrameCount;
        }
        LastFrameCrc = Crc;
        bHasLastFrameCrc = true;
    }

    Snapshot->RunSerial = RunSerial;
    Snapshot->FrameIndex = FrameCounter++;
    Snapshot->TimestampSeconds = NowSeconds();   // FIX: capture time, not completion time
    Snapshot->Width = DisplayRenderTarget->SizeX;
    Snapshot->Height = DisplayRenderTarget->SizeY;
    Snapshot->GroundTruth = GroundTruthUtils::ExtractBoxesFromStencilRenderTarget(
        GroundTruthRenderTarget,
        GroundTruthAssignments.StencilIDToClassID,
        bReadFullSilhouetteFromGreenChannel,
        bSaveGroundTruthFrames ? &Snapshot->GroundTruthPixels : nullptr);
    if (GroundTruthRenderTarget)
    {
        Snapshot->GroundTruthWidth = GroundTruthRenderTarget->SizeX;
        Snapshot->GroundTruthHeight = GroundTruthRenderTarget->SizeY;
    }
    Snapshot->WorldStateRows = CollectWorldStateRows(Snapshot->FrameIndex, Snapshot->TimestampSeconds);

    if (DepthRenderTarget)
    {
        ReadDepthCm(Snapshot->DepthCm, Snapshot->DepthWidth, Snapshot->DepthHeight);
    }

    Snapshot->bHasCameraPose = CaptureComponent.IsValid() && ForkliftActor != nullptr;
    if (Snapshot->bHasCameraPose)
    {
        Snapshot->CameraTransform = CaptureComponent->GetComponentTransform();
        Snapshot->ForkliftTransform = ForkliftActor->GetActorTransform();

        const USceneCaptureComponent2D* Capture2D = Cast<USceneCaptureComponent2D>(CaptureComponent.Get());
        Snapshot->HorizontalFOVDeg = Capture2D ? Capture2D->FOVAngle : FallbackHorizontalFOVDeg;
    }

    // ---- Tensor shapes (NCHW) ----
    TArray<uint32> Dims = {
        1,
        3,
        static_cast<uint32>(DisplayRenderTarget->SizeY),
        static_cast<uint32>(DisplayRenderTarget->SizeX)
    };
    TArray<UE::NNE::FTensorShape> InputShapes = { UE::NNE::FTensorShape::Make(Dims) };
    if (ModelInstance->SetInputTensorShapes(InputShapes) != UE::NNE::EResultStatus::Ok)
    {
        UE_LOG(LogTemp, Error, TEXT("YOLO: SetInputTensorShapes failed."));
        return;
    }

    // ---- Synchronous (evaluation) mode: the simulation waits for the result ----
    if (bSynchronousInference)
    {
        TArray<float> OutputTensorData;
        OutputTensorData.SetNum(1 * 84 * 8400);

        UE::NNE::FTensorBindingCPU InputBinding;
        InputBinding.Data = (void*)InputTensorData.GetData();
        InputBinding.SizeInBytes = InputTensorData.Num() * sizeof(float);

        UE::NNE::FTensorBindingCPU OutputBinding;
        OutputBinding.Data = OutputTensorData.GetData();
        OutputBinding.SizeInBytes = OutputTensorData.Num() * sizeof(float);

        const double InferenceStart = FPlatformTime::Seconds();
        auto RunStatus = ModelInstance->RunSync({ InputBinding }, { OutputBinding });
        const float InferenceTimeMs = static_cast<float>((FPlatformTime::Seconds() - InferenceStart) * 1000.0);

        if (RunStatus != decltype(RunStatus)::Ok)
        {
            UE_LOG(LogTemp, Warning, TEXT("YOLO: RunSync failed."));
            return;
        }

        ProcessInferenceResult(*Snapshot, OutputTensorData, InferenceTimeMs);
        return;
    }

    // ---- Asynchronous (real-time demo) mode ----
    bInferenceRunning = true;

    TWeakObjectPtr<AYOLOInferenceActor> WeakThis(this);
    TSharedPtr<UE::NNE::IModelInstanceGPU> LocalModelInstance = ModelInstance;

    AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask,
        [WeakThis, InputTensorData = MoveTemp(InputTensorData), LocalModelInstance, Snapshot]()
        {
            UE::NNE::FTensorBindingCPU InputBinding;
            InputBinding.Data = (void*)InputTensorData.GetData();
            InputBinding.SizeInBytes = InputTensorData.Num() * sizeof(float);

            TSharedPtr<TArray<float>> OutputTensorData = MakeShared<TArray<float>>();
            OutputTensorData->SetNum(1 * 84 * 8400);

            UE::NNE::FTensorBindingCPU OutputBinding;
            OutputBinding.Data = OutputTensorData->GetData();
            OutputBinding.SizeInBytes = OutputTensorData->Num() * sizeof(float);

            const double InferenceStart = FPlatformTime::Seconds();
            auto RunStatus = LocalModelInstance->RunSync({ InputBinding }, { OutputBinding });
            const float InferenceTimeMs = static_cast<float>((FPlatformTime::Seconds() - InferenceStart) * 1000.0);
            const bool bOk = (RunStatus == decltype(RunStatus)::Ok);

            AsyncTask(ENamedThreads::GameThread, [WeakThis, OutputTensorData, InferenceTimeMs, Snapshot, bOk]()
                {
                    if (!WeakThis.IsValid())
                    {
                        return;
                    }

                    if (bOk)
                    {
                        WeakThis->ProcessInferenceResult(*Snapshot, *OutputTensorData, InferenceTimeMs);
                    }
                    else
                    {
                        UE_LOG(LogTemp, Warning, TEXT("YOLO: async RunSync failed."));
                    }

                    if (Snapshot->RunSerial == WeakThis->RunSerial)
                    {
                        WeakThis->bInferenceRunning = false;
                    }
                });
        });
}

void AYOLOInferenceActor::ProcessInferenceResult(const FYOLOCaptureSnapshot& Snapshot, const TArray<float>& OutputTensor, float InferenceTimeMs)
{
    // Result belongs to a run that has already ended (async mode): drop it.
    if (Snapshot.RunSerial != RunSerial || !Logger.IsActive())
    {
        return;
    }

    const TArray<FBoundingBox2> Detections = BBoxMathUtils::ApplyNMS(ParseYOLOOutput(OutputTensor), NMSIoUThreshold);
    const TArray<FBoundingBox2> EvalDetections = BBoxMathUtils::FilterBoxesByClass(Detections, PersonClassID);

    // Person ground truth + index back into Snapshot.GroundTruth arrays.
    const FGroundTruthExtractionResult& GT = Snapshot.GroundTruth;
    TArray<FBoundingBox2> EvalGroundTruth;
    TArray<int32> EvalGTSourceIndex;
    for (int32 i = 0; i < GT.Boxes.Num(); ++i)
    {
        if (GT.Boxes[i].ClassID == PersonClassID)
        {
            EvalGroundTruth.Add(GT.Boxes[i]);
            EvalGTSourceIndex.Add(i);
        }
    }

    TArray<int32> DetectionToGT;
    TArray<float> DetectionIoU;

    FFrameRunStats Stats = BBoxMathUtils::EvaluateFrame(
        Snapshot.FrameIndex,
        Logger.GetRunId(),
        Logger.GetScenarioId(),
        static_cast<float>(Snapshot.TimestampSeconds),
        InferenceTimeMs,
        MatchIoUThreshold,
        EvalDetections,
        EvalGroundTruth,
        INDEX_NONE,
        &DetectionToGT,
        &DetectionIoU);

    const FString Prefix = Logger.RowPrefix(Snapshot.FrameIndex, Snapshot.TimestampSeconds);

    // ---- Distance estimation + closed loop ----
    USafetyBrakeController* Safety = SafetyController.Get();
    TArray<FPersonObservation> Observations;
    TArray<float> DepthsM;
    TArray<FVector> WorldPositionsCm;
    Observations.SetNum(EvalDetections.Num());
    DepthsM.Init(-1.0f, EvalDetections.Num());
    WorldPositionsCm.Init(FVector::ZeroVector, EvalDetections.Num());

    for (int32 d = 0; d < EvalDetections.Num(); ++d)
    {
        Observations[d] = EstimatePerson(EvalDetections[d], Snapshot, DepthsM[d], WorldPositionsCm[d]);
    }

    if (Safety)
    {
        Safety->OnPerception(Snapshot.TimestampSeconds, Observations);
    }

    // ---- detections.csv (person detections only) ----
    TArray<bool> GTDetected;
    GTDetected.Init(false, EvalGroundTruth.Num());

    for (int32 d = 0; d < EvalDetections.Num(); ++d)
    {
        const FBoundingBox2& Box = EvalDetections[d];
        int32 MatchedStencil = -1;
        if (DetectionToGT[d] != INDEX_NONE)
        {
            GTDetected[DetectionToGT[d]] = true;
            const int32 SourceIdx = EvalGTSourceIndex[DetectionToGT[d]];
            MatchedStencil = GT.BoxStencilIDs.IsValidIndex(SourceIdx) ? GT.BoxStencilIDs[SourceIdx] : -1;
        }

        const FPersonObservation& Obs = Observations[d];
        const bool bInZone = Safety ? Safety->IsInZone(Obs) : false;

        Logger.AddDetectionRow(Prefix + FString::Printf(
            TEXT("%d,%.5f,%.3f,%.3f,%.3f,%.3f,%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n"),
            Box.ClassID, Box.Confidence, Box.CenterX, Box.CenterY, Box.Width, Box.Height,
            MatchedStencil, DetectionIoU[d],
            DepthsM[d],
            Obs.bDepthValid ? WorldPositionsCm[d].X / 100.0 : 0.0,
            Obs.bDepthValid ? WorldPositionsCm[d].Y / 100.0 : 0.0,
            Obs.bDepthValid ? Obs.ForwardM : 0.0f,
            Obs.bDepthValid ? Obs.LateralM : 0.0f,
            bInZone ? 1 : 0));
    }

    // ---- ground_truth.csv ----
    for (int32 g = 0; g < EvalGroundTruth.Num(); ++g)
    {
        const int32 SourceIdx = EvalGTSourceIndex[g];
        const FBoundingBox2& Box = EvalGroundTruth[g];
        const int32 StencilID = GT.BoxStencilIDs.IsValidIndex(SourceIdx) ? GT.BoxStencilIDs[SourceIdx] : -1;
        const FName* ActorName = GroundTruthAssignments.StencilIDToActorName.Find(StencilID);

        Logger.AddGroundTruthRow(Prefix + FString::Printf(
            TEXT("%d,%s,%d,%.1f,%.1f,%.1f,%.1f,%d,%d,%d\n"),
            StencilID,
            ActorName ? *ActorName->ToString() : TEXT(""),
            Box.ClassID, Box.CenterX, Box.CenterY, Box.Width, Box.Height,
            GT.VisiblePixelCounts.IsValidIndex(SourceIdx) ? GT.VisiblePixelCounts[SourceIdx] : -1,
            GT.FullPixelCounts.IsValidIndex(SourceIdx) ? GT.FullPixelCounts[SourceIdx] : -1,
            GTDetected[g] ? 1 : 0));
    }

    // ---- frame_metrics.csv + world_state.csv ----
    Logger.AddFrameMetricsRow(Prefix + FString::Printf(
        TEXT("%.3f,%d,%d,%d,%d,%d,%.5f,%d,%d,%.3f,%d\n"),
        InferenceTimeMs, Stats.NumDetections, Stats.NumGroundTruth,
        Stats.NumTruePositives, Stats.NumFalsePositives, Stats.NumFalseNegatives, Stats.MeanMatchedIoU,
        (Safety && Safety->bLastFrameHazard) ? 1 : 0,
        Safety ? static_cast<int32>(Safety->GetState()) : -1,
        Safety ? Safety->GetTriggerDistanceM() : -1.0f,
        Snapshot.bStaleFrame ? 1 : 0));

    Logger.AddWorldStateRow(Snapshot.WorldStateRows);

    // ---- Images ----
    if (bSaveCleanFrames)
    {
        RunIOUtils::QueuePixelDataForSave(
            Snapshot.Pixels, Snapshot.Width, Snapshot.Height,
            Logger.GetFramesFolder() / FString::Printf(TEXT("frame_%05d.png"), Snapshot.FrameIndex));
    }

    if (bSaveDepthFrames && Snapshot.DepthCm.Num() > 0)
    {
        RunIOUtils::QueueDepthForSave(
            Snapshot.DepthCm, Snapshot.DepthWidth, Snapshot.DepthHeight, MaxValidDepthM * 100.0f,
            Logger.GetFramesFolder() / FString::Printf(TEXT("depth_%05d.png"), Snapshot.FrameIndex));
    }

    if (bSaveGroundTruthFrames && Snapshot.GroundTruthPixels.Num() > 0)
    {
        RunIOUtils::QueuePixelDataForSave(
            Snapshot.GroundTruthPixels, Snapshot.GroundTruthWidth, Snapshot.GroundTruthHeight,
            Logger.GetFramesFolder() / FString::Printf(TEXT("gt_%05d.png"), Snapshot.FrameIndex));
    }

    if (RunIOUtils::ShouldSaveSampleImage(Stats, bSaveSampleImages, bSaveFailureFramesOnly, ImageSaveInterval))
    {
        RunIOUtils::QueueAnnotatedPixelDataForSave(
            Snapshot.Pixels, Snapshot.Width, Snapshot.Height,
            Logger.GetSamplesFolder() / FString::Printf(TEXT("frame_%05d.png"), Snapshot.FrameIndex),
            EvalDetections, EvalGroundTruth,
            FColor(0, 255, 0, 255), FColor(255, 0, 0, 255), 2);
    }

    Logger.OnFrameLogged();

    UE_LOG(LogTemp, Log, TEXT("Frame %d | Det=%d GT=%d TP=%d FP=%d FN=%d | %.1f ms"),
        Snapshot.FrameIndex, Stats.NumDetections, Stats.NumGroundTruth,
        Stats.NumTruePositives, Stats.NumFalsePositives, Stats.NumFalseNegatives, InferenceTimeMs);

    OnInferenceCompleted(Detections, GT.Boxes);
}

// ============================================================================================
// Depth-based distance estimation (ideal depth sensor)
// ============================================================================================

bool AYOLOInferenceActor::ReadDepthCm(TArray<float>& OutDepth, int32& OutWidth, int32& OutHeight) const
{
    OutDepth.Reset();
    OutWidth = 0;
    OutHeight = 0;

    if (!DepthRenderTarget)
    {
        return false;
    }

    FTextureRenderTargetResource* RTResource = DepthRenderTarget->GameThread_GetRenderTargetResource();
    if (!RTResource)
    {
        return false;
    }

    // RCM_MinMax keeps the raw float values (depth in cm) instead of normalizing them to [0, 1].
    TArray<FLinearColor> Pixels;
    if (!RTResource->ReadLinearColorPixels(Pixels, FReadSurfaceDataFlags(RCM_MinMax, CubeFace_MAX)))
    {
        UE_LOG(LogTemp, Warning, TEXT("YOLO: ReadLinearColorPixels on DepthRenderTarget failed."));
        return false;
    }

    OutWidth = DepthRenderTarget->SizeX;
    OutHeight = DepthRenderTarget->SizeY;
    if (Pixels.Num() != OutWidth * OutHeight)
    {
        OutWidth = OutHeight = 0;
        return false;
    }

    OutDepth.SetNumUninitialized(Pixels.Num());
    for (int32 i = 0; i < Pixels.Num(); ++i)
    {
        OutDepth[i] = Pixels[i].R;
    }
    return true;
}

FPersonObservation AYOLOInferenceActor::EstimatePerson(const FBoundingBox2& Box, const FYOLOCaptureSnapshot& Snapshot,
    float& OutDepthM, FVector& OutWorldCm) const
{
    FPersonObservation Obs;
    Obs.Confidence = Box.Confidence;
    OutDepthM = -1.0f;
    OutWorldCm = FVector::ZeroVector;

    if (Snapshot.DepthCm.Num() == 0 || !Snapshot.bHasCameraPose || Snapshot.Width <= 0 || Snapshot.Height <= 0)
    {
        return Obs;
    }

    // Box (RGB pixel coords) -> depth image coords.
    const float ScaleX = static_cast<float>(Snapshot.DepthWidth) / Snapshot.Width;
    const float ScaleY = static_cast<float>(Snapshot.DepthHeight) / Snapshot.Height;

    // Central region of the box (40% of width, 50% of height) to avoid background pixels at the edges.
    const int32 X0 = FMath::Clamp(FMath::FloorToInt((Box.CenterX - 0.2f * Box.Width) * ScaleX), 0, Snapshot.DepthWidth - 1);
    const int32 X1 = FMath::Clamp(FMath::CeilToInt((Box.CenterX + 0.2f * Box.Width) * ScaleX), 0, Snapshot.DepthWidth - 1);
    const int32 Y0 = FMath::Clamp(FMath::FloorToInt((Box.CenterY - 0.25f * Box.Height) * ScaleY), 0, Snapshot.DepthHeight - 1);
    const int32 Y1 = FMath::Clamp(FMath::CeilToInt((Box.CenterY + 0.25f * Box.Height) * ScaleY), 0, Snapshot.DepthHeight - 1);

    const float MaxDepthCm = MaxValidDepthM * 100.0f;
    const float MinDepthCm = FMath::Max(1.0f, MinValidDepthM * 100.0f);
    TArray<float> Values;
    Values.Reserve((X1 - X0 + 1) * (Y1 - Y0 + 1));

    for (int32 Y = Y0; Y <= Y1; ++Y)
    {
        for (int32 X = X0; X <= X1; ++X)
        {
            const float D = Snapshot.DepthCm[Y * Snapshot.DepthWidth + X];
            if (D > MinDepthCm && D < MaxDepthCm)
            {
                Values.Add(D);
            }
        }
    }

    if (Values.Num() == 0)
    {
        return Obs;
    }

    Values.Sort();
    const int32 Index = FMath::Clamp(FMath::FloorToInt(DepthPercentile * (Values.Num() - 1)), 0, Values.Num() - 1);
    const float ZCm = Values[Index];   // planar depth along the optical axis

    // Pinhole back-projection (principal point = image center).
    const float Focal = (0.5f * Snapshot.Width) / FMath::Tan(FMath::DegreesToRadians(0.5f * Snapshot.HorizontalFOVDeg));
    const float RightCm = (Box.CenterX - 0.5f * Snapshot.Width) / Focal * ZCm;
    const float DownCm = (Box.CenterY - 0.5f * Snapshot.Height) / Focal * ZCm;

    // Unreal camera frame: X forward, Y right, Z up.
    const FVector CameraLocal(ZCm, RightCm, -DownCm);
    OutWorldCm = Snapshot.CameraTransform.TransformPositionNoScale(CameraLocal);

    const FVector InForklift = Snapshot.ForkliftTransform.InverseTransformPositionNoScale(OutWorldCm);
    const USafetyBrakeController* Safety = SafetyController.Get();
    const float FrontOffsetM = Safety ? Safety->FrontOffsetFromOriginM : 0.0f;

    OutDepthM = ZCm / 100.0f;
    Obs.bDepthValid = true;
    Obs.ForwardM = static_cast<float>(InForklift.X / 100.0) - FrontOffsetM;
    Obs.LateralM = static_cast<float>(InForklift.Y / 100.0);
    return Obs;
}