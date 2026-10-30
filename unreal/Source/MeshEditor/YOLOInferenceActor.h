#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Engine/Texture2D.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"  // CPU inference interface
#include "NNERuntimeGPU.h"
#include "NNEModelData.h"
#include "Async/AsyncWork.h"  // For async tasks
#include "Camera/CameraComponent.h"
#include "Types/DetectionTypes.h"
#include "Types/EvaluationTypes.h"
#include "Types/GroundTruthTypes.h"
#include "Utils/BBoxMathUtils.h"
#include "Utils/RunIOUtils.h"
#include "Utils/EvalRunLogger.h"
#include "SafetyBrakeController.h"
#include "YOLOInferenceActor.generated.h"

class USceneComponent;

/** Everything captured at the moment a frame is taken, so results are logged with capture-time data. */
struct FYOLOCaptureSnapshot
{
    int32 RunSerial = 0;
    int32 FrameIndex = 0;
    double TimestampSeconds = 0.0;   // game time at CAPTURE (not when inference finished)
    int32 Width = 0;
    int32 Height = 0;
    TArray<FColor> Pixels;
    FGroundTruthExtractionResult GroundTruth;
    TArray<FColor> GroundTruthPixels;  // raw stencil image, only filled when bSaveGroundTruthFrames
    int32 GroundTruthWidth = 0;
    int32 GroundTruthHeight = 0;
    FString WorldStateRows;          // pre-formatted world_state.csv rows for this frame

    // Closed loop / distance estimation (captured at the same moment)
    TArray<float> DepthCm;           // planar scene depth per pixel (cm), empty if no depth target
    int32 DepthWidth = 0;
    int32 DepthHeight = 0;
    bool bHasCameraPose = false;
    FTransform CameraTransform;
    FTransform ForkliftTransform;
    float HorizontalFOVDeg = 90.0f;

    /** Image identical to the previous frame while the forklift was moving (render target not updated). */
    bool bStaleFrame = false;
};

UCLASS(Blueprintable)
class MESHEDITOR_API AYOLOInferenceActor : public AActor
{
    GENERATED_BODY()

public:

    AYOLOInferenceActor();

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    UFUNCTION(BlueprintCallable, Category = "YOLO Inference")
    void RunYOLOInference();

    /** Starts a new logged run (new output folder). Called by the ScenarioManager for every scenario repetition. */
    UFUNCTION(BlueprintCallable, Category = "YOLO Inference|Evaluation")
    void BeginEvaluationRun(const FString& BatchId, const FString& ScenarioId, int32 Repetition, AActor* InForklift);

    /**
     * Flushes all CSVs and writes run_metadata.json.
     * ExtraMetadataJsonFields: extra JSON key/value pairs without surrounding braces,
     * e.g. "\"end_reason\": \"RouteCompleted\", \"seed\": 42"
     */
    UFUNCTION(BlueprintCallable, Category = "YOLO Inference|Evaluation")
    void EndEvaluationRun(const FString& ExtraMetadataJsonFields);

    /**
     * Set by the ScenarioManager for the active run: returns its metadata fields, so a run that is cut short
     * (play stopped) still gets the full run_metadata.json. Cleared when the run ends.
     */
    TFunction<FString()> RunMetadataProvider;

    UFUNCTION(BlueprintImplementableEvent, Category = "YOLO Inference")
    void OnInferenceCompleted(const TArray<FBoundingBox2>& Detections, const TArray<FBoundingBox2>& GroundTruthBoxes);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference")
    UCameraComponent* CameraComponent = nullptr;

    UFUNCTION(BlueprintCallable, Category = "YOLO Inference")
    bool CopyToDisplayRenderTarget();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference UI")
    UMaterialInterface* CopyTextureMaterial = nullptr;

    UPROPERTY(Transient)
    UMaterialInstanceDynamic* CopyTextureMID = nullptr;

    // ---------------- Model ----------------

    /** NNE model asset path. Change this to compare yolo11n / s / m / x. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Model")
    FString ModelAssetPath = TEXT("/Game/Models/yolo11x.yolo11x");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Model")
    float ConfidenceThreshold = 0.25f;

    /** 0.7 = Ultralytics default. The previous hard-coded value was 0.45. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Model")
    float NMSIoUThreshold = 0.7f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Model")
    float MatchIoUThreshold = 0.5f;

    /** COCO class 0 = person. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Model")
    int32 PersonClassID = 0;

    // ---------------- Evaluation ----------------

    /**
     * Evaluation mode: run inference on the game thread so the simulation waits for it.
     * Combine with a fixed frame rate (Project Settings > General Settings > Framerate)
     * for reproducible runs. Latency is still measured and logged (inference_ms).
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Evaluation")
    bool bSynchronousInference = true;

    /** Save every evaluated frame as a clean PNG (no boxes) for offline evaluation in Python. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Evaluation")
    bool bSaveCleanFrames = true;

    /**
     * Calls CaptureScene() on the RGB, ground-truth and depth capture components right before each inference,
     * so the frame is always rendered for this moment. Without it, captures only update when the editor
     * viewport renders (e.g. not while the PIE window is in the background), and stale images are evaluated.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Evaluation")
    bool bForceSceneCaptureBeforeInference = true;

    /**
     * Save the depth frame next to each clean frame (frames/depth_NNNNN.png): 16-bit grayscale PNG, planar depth
     * along the camera axis in millimeters, 0 = invalid or beyond MaxValidDepthM. Lets the Python replay estimate
     * distances for any detector. Needs DepthRenderTarget; switch off for closed-loop batches together with clean frames.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Evaluation")
    bool bSaveDepthFrames = true;

    /**
    * Save the raw ground-truth stencil image next to each clean frame (frames/gt_NNNNN.png):
    * R = visible stencil ID, G = full-silhouette stencil ID, 0 = no worker. For figures; off by default.
    */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Evaluation")
    bool bSaveGroundTruthFrames = false;

    /** Set once the stencil material writes the full silhouette (no depth test) into the G channel. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Evaluation")
    bool bReadFullSilhouetteFromGreenChannel = false;

    // ---------------- Depth / closed loop ----------------

    /**
     * Depth of the camera view: a second SceneCaptureComponent2D on the forklift with the SAME transform
     * and FOV as the RGB capture, Capture Source = "SceneDepth in R", render target format R32f (640x640).
     * Used to estimate each detected person's distance (ideal depth sensor) and to drive the
     * SafetyBrakeController. Leave empty to disable distance estimation.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Depth")
    UTextureRenderTarget2D* DepthRenderTarget = nullptr;

    /** Depth values beyond this are treated as invalid (sky, far background). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Depth")
    float MaxValidDepthM = 60.0f;

    /**
     * Depth values nearer than this are ignored when estimating a person's distance. Set it a little beyond the
     * fork tips (about 1.8 m from the camera) so pixels of the forklift itself or of a load on the forks are never
     * taken as the person's distance. 0 = off. Saved depth PNGs are not affected.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Depth", meta = (ClampMin = "0.0"))
    float MinValidDepthM = 0.0f;

    /** Percentile of the depths in the central part of the box (low = nearest surface = the person, not the background). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Depth", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float DepthPercentile = 0.25f;

    /** Used only if the capture component's FOV can't be read. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference|Depth")
    float FallbackHorizontalFOVDeg = 90.0f;

    /** Forklift used for world-state logging when running without the ScenarioManager. */
    UPROPERTY(EditInstanceOnly, BlueprintReadWrite, Category = "YOLO Inference|Evaluation")
    TObjectPtr<AActor> ForkliftActor = nullptr;

    // ---------------- Ground truth ----------------

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Ground Truth")
    UTextureRenderTarget2D* GroundTruthRenderTarget = nullptr;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Ground Truth")
    FName GroundTruthActorTag = TEXT("GroundTruthPerson");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Ground Truth")
    int32 GroundTruthClassID = 0;

    FGroundTruthAssignmentResult GroundTruthAssignments;

    // ---------------- Output ----------------

    /** Annotated (boxes drawn) sample images, e.g. failure frames. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Run Output")
    bool bSaveSampleImages = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Run Output")
    bool bSaveFailureFramesOnly = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Run Output", meta = (ClampMin = "0"))
    int32 ImageSaveInterval = 0;

protected:
    bool LoadModel();
    bool PreprocessRenderTarget(UTextureRenderTarget2D* RenderTarget, TArray<float>& TensorData, TArray<FColor>* OutPixels);
    TArray<FBoundingBox2> ParseYOLOOutput(const TArray<float>& OutputData);

    void SaveTensorToRaw(const TArray<float>& Tensor, const FString& FilePath);

    TSharedPtr<UE::NNE::IModelInstanceGPU> ModelInstance;

    UPROPERTY()
    TObjectPtr<UNNEModelData> ModelData;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference")
    UTextureRenderTarget2D* InputRenderTarget;

    UPROPERTY(BlueprintReadOnly, Category = "YOLO Inference")
    UTexture2D* LastInputTexture = nullptr;

    UTexture2D* CreateTextureFromPixels(const TArray<FColor>& Pixels, int32 Width, int32 Height);

    bool bInferenceRunning = false;

    int32 FrameCounter = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "YOLO Inference")
    UTextureRenderTarget2D* DisplayRenderTarget;

private:
    void ProcessInferenceResult(const FYOLOCaptureSnapshot& Snapshot, const TArray<float>& OutputTensor, float InferenceTimeMs);
    FString CollectWorldStateRows(int32 FrameIndex, double TimestampSeconds) const;
    USceneComponent* FindCaptureComponent() const;
    void EnsureRunActive();
    bool ReadDepthCm(TArray<float>& OutDepth, int32& OutWidth, int32& OutHeight) const;
    FPersonObservation EstimatePerson(const FBoundingBox2& Box, const FYOLOCaptureSnapshot& Snapshot,
        float& OutDepthM, FVector& OutWorldCm) const;
    double NowSeconds() const;
    void ForceSceneCaptures();

    TArray<TWeakObjectPtr<class USceneCaptureComponent2D>> ForcedCaptureComponents;
    uint32 LastFrameCrc = 0;
    bool bHasLastFrameCrc = false;
    int32 StaleFrameCount = 0;

    FEvalRunLogger Logger;
    int32 RunSerial = 0;
    double RunStartTimeSeconds = 0.0;
    TWeakObjectPtr<USceneComponent> CaptureComponent;
    TWeakObjectPtr<USafetyBrakeController> SafetyController;
};
