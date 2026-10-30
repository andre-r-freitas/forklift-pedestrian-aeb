#include "ScenarioManager.h"
#include "YOLOInferenceActor.h"
#include "PedestrianCharacter.h"
#include "ForkliftRoute.h"
#include "ForkliftRouteFollower.h"
#include "SafetyBrakeController.h"
#include "GameFramework/Actor.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Logging/LogMacros.h"
#include "Components/PrimitiveComponent.h"
#include "Engine/Engine.h"
#include "Engine/EngineTypes.h"
#include "Engine/TargetPoint.h"
#include "EngineUtils.h"
#include "Math/RandomStream.h"
#include "Components/SplineComponent.h"
#include "Misc/DateTime.h"
#include "TimerManager.h"

AScenarioManager::AScenarioManager()
{
    PrimaryActorTick.bCanEverTick = true;
}

void AScenarioManager::BeginPlay()
{
    Super::BeginPlay();

    InitialPedestrianTransforms.Reset();
    for (AActor* Pedestrian : Pedestrians)
    {
        InitialPedestrianTransforms.Add(IsValid(Pedestrian) ? Pedestrian->GetActorTransform() : FTransform::Identity);
    }

    ScenarioState = EScenarioState::Idle;

    // In batch mode ScenarioConfig may be left empty.
    if (!IsValid(ScenarioConfig) && BatchScenarios.Num() > 0)
    {
        ScenarioConfig = BatchScenarios[0];
    }

    if (!IsValid(ScenarioConfig))
    {
        UE_LOG(LogTemp, Error, TEXT("ScenarioManager: No ScenarioConfig (or BatchScenarios) assigned."));
        ScenarioState = EScenarioState::Failed;
        return;
    }

    if (!IsValid(Forklift))
    {
        UE_LOG(LogTemp, Error, TEXT("ScenarioManager: No Forklift actor assigned."));
        ScenarioState = EScenarioState::Failed;
        return;
    }

    if (Pedestrians.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("ScenarioManager: No pedestrian actors assigned."));
    }

    UE_LOG(LogTemp, Log, TEXT("ScenarioManager ready. Scenario: %s, batch entries: %d"),
        *GetScenarioIdString(), BatchScenarios.Num());

    ScenarioState = EScenarioState::Ready;

    if (bRunBatchOnBeginPlay && BatchScenarios.Num() > 0)
    {
        // Next tick, so every other actor (inference actor, stencil assignment) has run BeginPlay.
        GetWorldTimerManager().SetTimerForNextTick(this, &AScenarioManager::StartBatch);
    }
}

void AScenarioManager::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);

    if (ScenarioState == EScenarioState::WaitingForSpeed)
    {
        WaitForSpeedElapsed += DeltaSeconds;

        const float TargetKmh = CurrentSpeedKmh;
        const bool bAtSpeed = GetForkliftForwardSpeedKmh() >= AtSpeedFraction * TargetKmh;

        if (bAtSpeed || WaitForSpeedElapsed >= MaxWaitForSpeedSeconds)
        {
            bSpeedWaitTimedOut = !bAtSpeed;
            if (bSpeedWaitTimedOut)
            {
                UE_LOG(LogTemp, Warning,
                    TEXT("ScenarioManager: forklift did not reach %.0f%% of %.1f km/h within %.1f s (now %.1f km/h). Starting capture anyway."),
                    AtSpeedFraction * 100.0f, TargetKmh, MaxWaitForSpeedSeconds, GetForkliftForwardSpeedKmh());
            }
            BeginScenarioCapture();
        }
        return;
    }

    if (ScenarioState != EScenarioState::Running)
    {
        return;
    }

    ScenarioElapsedTime += DeltaSeconds;

    UpdatePendingPedestrianStarts();

    if (HasForkliftReachedRouteEnd() && IsForkliftStopped())
    {
        ScenarioEndReason = EScenarioEndReason::RouteCompleted;
        FinishScenario();
        return;
    }

    if (bEnableSafetyBraking && IsValid(Forklift))
    {
        if (const USafetyBrakeController* Safety = Forklift->FindComponentByClass<USafetyBrakeController>())
        {
            const double StoppedAt = Safety->GetStoppedTime();
            if (Safety->GetState() == ESafetyBrakeState::Stopped && StoppedAt >= 0.0
                && GetWorld()->GetTimeSeconds() - StoppedAt >= SafetyStopEndDelayS)
            {
                ScenarioEndReason = EScenarioEndReason::SafetyStop;
                FinishScenario();
                return;
            }
        }
    }

    if (ScenarioElapsedTime >= ScenarioConfig->DurationSeconds)
    {
        ScenarioEndReason = EScenarioEndReason::DurationExpired;
        FinishScenario();
        return;
    }
}

// ============================================================================================
// Batch
// ============================================================================================

void AScenarioManager::StartBatch()
{
    BuildBatchRuns();

    if (BatchRuns.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("ScenarioManager: StartBatch called with no valid BatchScenarios."));
        return;
    }

    // Number every run 1..N in the full list, then apply the resume / re-run selection. Seeds and folder names come
    // from the spec, so a run is identical whether it runs in the original batch or in a resumed one.
    BatchRunTotal = BatchRuns.Num();
    for (int32 i = 0; i < BatchRuns.Num(); ++i)
    {
        BatchRuns[i].RunNumber = i + 1;
    }
    if (OnlyRunNumbers.Num() > 0)
    {
        BatchRuns.RemoveAll([this](const FBatchRunSpec& R) { return !OnlyRunNumbers.Contains(R.RunNumber); });
    }
    else if (ResumeFromRun > 1)
    {
        BatchRuns.RemoveAll([this](const FBatchRunSpec& R) { return R.RunNumber < ResumeFromRun; });
    }
    if (BatchRuns.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("ScenarioManager: ResumeFromRun / OnlyRunNumbers selected no runs (batch has %d)."), BatchRunTotal);
        return;
    }

    bBatchActive = true;
    BatchRunIndex = 0;
    CurrentBatchId = FString::Printf(TEXT("Batch_%s"), *FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S")));

    UE_LOG(LogTemp, Warning, TEXT("ScenarioManager: batch %s started: %d of %d runs (%d scenario assets, variants expanded, %d repetitions each%s)."),
        *CurrentBatchId, BatchRuns.Num(), BatchRunTotal, BatchScenarios.Num(), RepetitionsPerScenario,
        OnlyRunNumbers.Num() > 0 ? TEXT(", selected run numbers only") : (ResumeFromRun > 1 ? TEXT(", resumed") : TEXT("")));

    StartCurrentBatchRun();
}

void AScenarioManager::BuildBatchRuns()
{
    BatchRuns.Reset();
    int32 VariantCounter = 0;

    for (int32 ScenarioIdx = 0; ScenarioIdx < BatchScenarios.Num(); ++ScenarioIdx)
    {
        const UScenarioConfigDataAsset* Config = BatchScenarios[ScenarioIdx];
        if (!IsValid(Config))
        {
            UE_LOG(LogTemp, Error, TEXT("ScenarioManager: batch entry %d is empty, skipping it."), ScenarioIdx);
            continue;
        }

        const bool bSpeedVariants = Config->VariantForkliftSpeedsKmh.Num() > 0;
        const bool bTimingVariants = Config->VariantTimingShiftsS.Num() > 0;

        TArray<float> Speeds = bSpeedVariants ? Config->VariantForkliftSpeedsKmh : TArray<float>{ Config->ForkliftSpeedKmh };
        TArray<float> Shifts = bTimingVariants ? Config->VariantTimingShiftsS : TArray<float>{ 0.0f };

        for (const float Speed : Speeds)
        {
            for (const float Shift : Shifts)
            {
                FString Suffix;
                if (bSpeedVariants)
                {
                    Suffix += FString::Printf(TEXT("_v%.0f"), Speed);
                }
                if (bTimingVariants)
                {
                    Suffix += FString::Printf(TEXT("_t%s%.1f"), Shift >= 0.0f ? TEXT("+") : TEXT(""), Shift);
                }

                for (int32 Rep = 0; Rep < RepetitionsPerScenario; ++Rep)
                {
                    FBatchRunSpec Spec;
                    Spec.ScenarioIndex = ScenarioIdx;
                    Spec.VariantIndex = VariantCounter;
                    Spec.SpeedKmh = Speed;
                    Spec.TimingShiftS = Shift;
                    Spec.Suffix = Suffix;
                    Spec.Repetition = Rep;
                    BatchRuns.Add(Spec);
                }
                ++VariantCounter;
            }
        }
    }
}

void AScenarioManager::StopBatch()
{
    bBatchActive = false;
    GetWorldTimerManager().ClearTimer(NextRunTimerHandle);

    if (ScenarioState == EScenarioState::Running)
    {
        ScenarioEndReason = EScenarioEndReason::Stopped;
        FinishScenario();
    }
}

void AScenarioManager::StartCurrentBatchRun()
{
    if (!bBatchActive || !BatchRuns.IsValidIndex(BatchRunIndex))
    {
        return;
    }

    const FBatchRunSpec& Spec = BatchRuns[BatchRunIndex];
    ScenarioConfig = BatchScenarios[Spec.ScenarioIndex];
    CurrentSpeedKmh = Spec.SpeedKmh;
    CurrentTimingShiftS = Spec.TimingShiftS;
    CurrentVariantSuffix = Spec.Suffix;
    CurrentRepetition = Spec.Repetition;
    // Common random numbers: the seed ignores the speed / timing variant, so repetition N draws the SAME pedestrian
    // speed and timing jitter at every forklift speed and timing shift. Differences between variants then come from
    // the variant itself, not from different random draws.
    CurrentSeed = BaseSeed + 1000 * Spec.ScenarioIndex + Spec.Repetition;
    bApplyVariationThisRun = true;

    CurrentBatchRunNumber = Spec.RunNumber;

    UE_LOG(LogTemp, Warning, TEXT("ScenarioManager: batch run %d/%d: %s rep %d (seed %d)"),
        Spec.RunNumber, BatchRunTotal, *GetScenarioIdString(), CurrentRepetition, CurrentSeed);

    StartScenario();

    if (ScenarioState == EScenarioState::Failed)
    {
        UE_LOG(LogTemp, Error, TEXT("ScenarioManager: run failed to start, moving on."));
        AdvanceBatch();
    }
}

void AScenarioManager::AdvanceBatch()
{
    ++BatchRunIndex;

    if (BatchRuns.IsValidIndex(BatchRunIndex))
    {
        GetWorldTimerManager().SetTimer(
            NextRunTimerHandle, this, &AScenarioManager::StartCurrentBatchRun,
            FMath::Max(0.01f, DelayBetweenRunsSeconds), false);
    }
    else
    {
        bBatchActive = false;
        UE_LOG(LogTemp, Warning, TEXT("ScenarioManager: batch %s finished."), *CurrentBatchId);
        if (GEngine)
        {
            GEngine->AddOnScreenDebugMessage(-1, 10.0f, FColor::Green, FString::Printf(TEXT("Batch %s finished"), *CurrentBatchId));
        }
    }
}

// ============================================================================================
// Single run
// ============================================================================================

void AScenarioManager::StartScenario()
{
    if (ScenarioState == EScenarioState::Running ||
        ScenarioState == EScenarioState::WarmingUp ||
        ScenarioState == EScenarioState::WaitingForSpeed)
    {
        UE_LOG(LogTemp, Warning, TEXT("ScenarioManager: Scenario is already running."));
        return;
    }

    if (!IsValid(ScenarioConfig))
    {
        UE_LOG(LogTemp, Error, TEXT("ScenarioManager: No valid ScenarioConfig."));
        ScenarioState = EScenarioState::Failed;
        return;
    }

    if (!IsValid(Forklift))
    {
        UE_LOG(LogTemp, Error, TEXT("ScenarioManager: No valid Forklift."));
        ScenarioState = EScenarioState::Failed;
        return;
    }

    if (!bBatchActive)
    {
        // Single run: its own batch folder, no randomisation.
        CurrentBatchId = FString::Printf(TEXT("Single_%s"), *FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S")));
        CurrentRepetition = 0;
        CurrentSeed = BaseSeed;
        bApplyVariationThisRun = false;
        CurrentSpeedKmh = ScenarioConfig->ForkliftSpeedKmh;
        CurrentTimingShiftS = 0.0f;
        CurrentVariantSuffix.Reset();
    }

    ResetScenario();

    if (ScenarioState == EScenarioState::Failed)
    {
        UE_LOG(LogTemp, Error, TEXT("ScenarioManager: Scenario reset failed. Scenario will not start."));
        return;
    }

    ScenarioElapsedTime = 0.0f;
    ScenarioEndReason = EScenarioEndReason::None;
    ScenarioState = EScenarioState::WarmingUp;

    UE_LOG(LogTemp, Log, TEXT("ScenarioManager: Warming up for %.2f seconds."), ScenarioConfig->WarmupTimeSeconds);

    GetWorldTimerManager().SetTimer(
        WarmupTimerHandle,
        this,
        &AScenarioManager::OnWarmupFinished,
        FMath::Max(0.01f, ScenarioConfig->WarmupTimeSeconds),
        false);
}

void AScenarioManager::OnWarmupFinished()
{
    if (ScenarioState != EScenarioState::WarmingUp)
    {
        return;
    }

    WaitForSpeedElapsed = 0.0f;
    bSpeedWaitTimedOut = false;

    // No route or no speed: nothing to wait for.
    if (!IsValid(ForkliftRoute) || CurrentSpeedKmh <= 0.0f)
    {
        BeginScenarioCapture();
        return;
    }

    StartForklift();
    ScenarioState = EScenarioState::WaitingForSpeed;
}

void AScenarioManager::StartForklift()
{
    if (!IsValid(Forklift) || !IsValid(ForkliftRoute))
    {
        return;
    }

    UForkliftRouteFollower* RouteFollower = Forklift->FindComponentByClass<UForkliftRouteFollower>();
    if (!RouteFollower)
    {
        UE_LOG(LogTemp, Warning, TEXT("ScenarioManager: Forklift has no UForkliftRouteFollower component."));
        return;
    }

    RouteFollower->StartFollowing(ForkliftRoute->GetSpline(), CurrentSpeedKmh);
}

void AScenarioManager::ResetScenario()
{
    GetWorldTimerManager().ClearTimer(WarmupTimerHandle);
    GetWorldTimerManager().ClearTimer(InferenceTimerHandle);
    ClearPedestrianStartTimers();

    if (!IsValid(ScenarioConfig))
    {
        UE_LOG(
            LogTemp,
            Error,
            TEXT("ScenarioManager: Cannot reset without a ScenarioConfig.")
        );

        ScenarioState = EScenarioState::Failed;
        return;
    }

    if (!ResolveScenarioConfiguration())
    {
        UE_LOG(
            LogTemp,
            Error,
            TEXT(
                "ScenarioManager: Could not resolve pedestrian TargetPoints."
            )
        );

        ScenarioState = EScenarioState::Failed;
        return;
    }

    // Batch repetitions: randomise pedestrian speed and start delay (reproducible via seed).
    ApplyRepetitionVariation();

    if (!ResolveForkliftRoute())
    {
        UE_LOG(
            LogTemp,
            Error,
            TEXT(
                "ScenarioManager: Could not resolve "
                "the forklift route."
            )
        );

        ScenarioState = EScenarioState::Failed;
        return;
    }

    if (!ResolveForkliftStartTransform())
    {
        UE_LOG(
            LogTemp,
            Error,
            TEXT(
                "ScenarioManager: Could not resolve "
                "the forklift start transform."
            )
        );

        ScenarioState = EScenarioState::Failed;
        return;
    }

    ScenarioState = EScenarioState::Resetting;

    /*
     * Reset the forklift.
     *
     * Moves the forklift using a physics-aware teleport.
     * Clears linear velocity.
     * Clears angular velocity.
     */
    if (IsValid(Forklift))
    {
        const FTransform ResetTransform(
            ResolvedForkliftStartRotation,
            ResolvedForkliftStartLocation
        );

        Forklift->SetActorTransform(
            ResetTransform,
            false,
            nullptr,
            ETeleportType::TeleportPhysics
        );

        UPrimitiveComponent* RootPrimitive =
            Cast<UPrimitiveComponent>(Forklift->GetRootComponent());

        if (RootPrimitive && RootPrimitive->IsSimulatingPhysics())
        {
            RootPrimitive->SetPhysicsLinearVelocity(
                FVector::ZeroVector
            );

            RootPrimitive->SetPhysicsAngularVelocityInDegrees(
                FVector::ZeroVector
            );
        }

        // Stop and reset the route follower on the forklift
        UForkliftRouteFollower* RouteFollower =
            Forklift->FindComponentByClass<UForkliftRouteFollower>();

        if (RouteFollower)
        {
            RouteFollower->StopFollowing();
            RouteFollower->ResetFollower();
        }

        // Closed loop: clear the braking controller and apply this run's on/off setting.
        if (USafetyBrakeController* Safety = Forklift->FindComponentByClass<USafetyBrakeController>())
        {
            Safety->ResetController();
            Safety->bEnabled = bEnableSafetyBraking;
        }
        else if (bEnableSafetyBraking)
        {
            UE_LOG(LogTemp, Warning, TEXT("ScenarioManager: bEnableSafetyBraking is on, but the forklift has no USafetyBrakeController."));
        }

        UE_LOG(
            LogTemp,
            Log,
            TEXT(
                "ScenarioManager: Forklift reset to %s"
            ),
            *ResolvedForkliftStartLocation.ToString()
        );
        UE_LOG(
            LogTemp,
            Log,
            TEXT(
                "Forklift transform after reset: Location=%s Rotation=%s"
            ),
            *Forklift->GetActorLocation().ToString(),
            *Forklift->GetActorRotation().ToString()
        );
    }
    else
    {
        UE_LOG(
            LogTemp,
            Warning,
            TEXT("ScenarioManager: Forklift is invalid during reset.")
        );
    }

    /*
     * Reset pedestrians via helper
     */
    const int32 PedestrianCountToConfigure = FMath::Min(
        Pedestrians.Num(),
        ResolvedPedestrianConfigs.Num()
    );

    for (int32 Index = 0; Index < PedestrianCountToConfigure; ++Index)
    {
        AActor* Pedestrian = Pedestrians[Index];

        if (!IsValid(Pedestrian))
        {
            UE_LOG(
                LogTemp,
                Warning,
                TEXT("ScenarioManager: Pedestrian %d is invalid."),
                Index
            );

            continue;
        }

        const FScenarioPedestrianConfig& PedestrianConfig =
            ResolvedPedestrianConfigs[Index];

        ActivatePedestrian(Pedestrian);   // may have been parked by a previous scenario

        ResetPedestrianActor(
            Pedestrian,
            PedestrianConfig
        );
    }

    ParkUnusedPedestrians(PedestrianCountToConfigure);

    if (Pedestrians.Num() != ScenarioConfig->Pedestrians.Num())
    {
        UE_LOG(
            LogTemp,
            Warning,
            TEXT(
                "ScenarioManager: Actor/config pedestrian count mismatch. "
                "Actors: %d, Config entries: %d."
            ),
            Pedestrians.Num(),
            ScenarioConfig->Pedestrians.Num()
        );
    }

    /*
     * Enable or disable the configured occluder.
     *
     * The actual OccluderActor reference is stored on the ScenarioManager,
     * because it refers to an actor placed in the level.
     */
    TArray<AActor*> AllOccluders;
    if (IsValid(OccluderActor))
    {
        AllOccluders.AddUnique(OccluderActor.Get());
    }
    for (const TObjectPtr<AActor>& Occ : OccluderActors)
    {
        if (IsValid(Occ))
        {
            AllOccluders.AddUnique(Occ.Get());
        }
    }

    ActiveOccluderNames.Reset();
    for (AActor* Occ : AllOccluders)
    {
        const bool bShow = ScenarioConfig->bUseOccluder &&
            (ScenarioConfig->OccluderActorTag.IsNone() || Occ->ActorHasTag(ScenarioConfig->OccluderActorTag));

        Occ->SetActorHiddenInGame(!bShow);
        Occ->SetActorEnableCollision(bShow);   // set the load's collision preset to NoCollision: then this does nothing

        if (bShow)
        {
            if (!ActiveOccluderNames.IsEmpty())
            {
                ActiveOccluderNames += TEXT(";");
            }
            ActiveOccluderNames += Occ->GetName();
        }
    }

    if (ScenarioConfig->bUseOccluder && ActiveOccluderNames.IsEmpty())
    {
        UE_LOG(
            LogTemp,
            Warning,
            TEXT("ScenarioManager: Scenario requests an occluder (tag '%s'), but no assigned occluder matches."),
            *ScenarioConfig->OccluderActorTag.ToString()
        );
    }

    ScenarioState = EScenarioState::Ready;

    UE_LOG(
        LogTemp,
        Log,
        TEXT("Scenario actors reset.")
    );
}

void AScenarioManager::BeginScenarioCapture()
{
    if (!IsValid(ScenarioConfig))
    {
        ScenarioState = EScenarioState::Failed;
        return;
    }

    if (!IsValid(InferenceActor))
    {
        UE_LOG(LogTemp, Error, TEXT("ScenarioManager: No InferenceActor assigned."));
        ScenarioState = EScenarioState::Failed;
        return;
    }

    ScenarioState = EScenarioState::Running;
    ScenarioElapsedTime = 0.0f;
    ForkliftSpeedAtCaptureKmh = GetForkliftForwardSpeedKmh();

    // New output folder for this run (scenario id + repetition).
    InferenceActor->BeginEvaluationRun(CurrentBatchId, GetScenarioIdString(), CurrentRepetition, Forklift);

    // If play is stopped mid-run, the inference actor still writes the FULL metadata (end reason Aborted).
    TWeakObjectPtr<AScenarioManager> WeakThis(this);
    InferenceActor->RunMetadataProvider = [WeakThis]() -> FString
        {
            AScenarioManager* Manager = WeakThis.Get();
            if (!Manager)
            {
                return TEXT("\"end_reason\": \"Aborted\"");
            }
            if (Manager->ScenarioEndReason == EScenarioEndReason::None)
            {
                Manager->ScenarioEndReason = EScenarioEndReason::Aborted;
            }
            return Manager->BuildRunMetadataFields();
        };

    // Start pedestrians: by time delay (from capture start) or by forklift distance to their conflict point.
    ClearPedestrianStartTimers();
    PendingPedestrianStarts.Reset();

    const int32 PedestrianCountToStart = FMath::Min(Pedestrians.Num(), ResolvedPedestrianConfigs.Num());
    PedestrianTriggerDistanceM.Init(-1.0f, PedestrianCountToStart);
    PedestrianStartTimeS.Init(-1.0f, PedestrianCountToStart);
    PedestrianForkliftDistanceAtStartM.Init(-1.0f, PedestrianCountToStart);

    for (int32 Index = 0; Index < PedestrianCountToStart; ++Index)
    {
        AActor* Pedestrian = Pedestrians[Index];

        if (!IsValid(Pedestrian))
        {
            continue;
        }

        const FScenarioPedestrianConfig& PedestrianConfig = ResolvedPedestrianConfigs[Index];

        if (PedestrianConfig.bIsStationary)
        {
            continue;
        }

        const bool bDistanceTriggered =
            PedestrianConfig.StartMode != EPedestrianStartMode::Delay && IsValid(ForkliftRoute);

        if (bDistanceTriggered)
        {
            FPendingPedestrianStart Pending;
            Pending.PedestrianIndex = Index;
            Pending.TriggerDistanceM = ComputeTriggerDistanceM(PedestrianConfig);
            Pending.ConflictLocation = GetConflictLocation(PedestrianConfig);
            PedestrianTriggerDistanceM[Index] = Pending.TriggerDistanceM;
            PendingPedestrianStarts.Add(Pending);

            UE_LOG(LogTemp, Log, TEXT("ScenarioManager: pedestrian %d starts when forklift front is %.2f m from its conflict point."),
                Index, Pending.TriggerDistanceM);
            continue;
        }

        if (PedestrianConfig.StartMode != EPedestrianStartMode::Delay)
        {
            UE_LOG(LogTemp, Warning, TEXT("ScenarioManager: pedestrian %d uses a distance-based start but there is no forklift route; starting immediately."), Index);
        }

        if (PedestrianConfig.StartMode == EPedestrianStartMode::Delay && PedestrianConfig.StartDelaySeconds > 0.0f)
        {
            // Handles are stored so a pending start can be cancelled on reset/finish.
            FTimerHandle& DelayHandle = PedestrianStartTimers.AddDefaulted_GetRef();

            GetWorldTimerManager().SetTimer(
                DelayHandle,
                [this, Pedestrian, PedestrianConfig, Index]()
                {
                    if (IsValid(Pedestrian) && ScenarioState == EScenarioState::Running)
                    {
                        StartPedestrianActor(Pedestrian, PedestrianConfig);
                        if (PedestrianStartTimeS.IsValidIndex(Index))
                        {
                            PedestrianStartTimeS[Index] = ScenarioElapsedTime;
                        }
                    }
                },
                PedestrianConfig.StartDelaySeconds,
                false);
        }
        else
        {
            StartPedestrianActor(Pedestrian, PedestrianConfig);
            PedestrianStartTimeS[Index] = 0.0f;
        }
    }

    // Distance-triggered pedestrians whose trigger is already reached start right away.
    UpdatePendingPedestrianStarts();

    // Forklift is normally already driving (started in OnWarmupFinished).
    if (!IsValid(ForkliftRoute))
    {
        UE_LOG(LogTemp, Log, TEXT("ScenarioManager: No forklift route selected. Forklift will remain stationary."));
    }

    UE_LOG(LogTemp, Log, TEXT("ScenarioManager: Scenario capture started: %s rep %d (forklift at %.1f km/h)"),
        *GetScenarioIdString(), CurrentRepetition, ForkliftSpeedAtCaptureKmh);

    // Capture the first frame immediately, then at a fixed interval.
    RunScenarioInference();

    GetWorldTimerManager().SetTimer(
        InferenceTimerHandle,
        this,
        &AScenarioManager::RunScenarioInference,
        InferenceIntervalSeconds,
        true);
}

void AScenarioManager::RunScenarioInference()
{
    if (ScenarioState != EScenarioState::Running)
    {
        return;
    }

    if (!IsValid(InferenceActor))
    {
        UE_LOG(LogTemp, Error, TEXT("ScenarioManager: InferenceActor became invalid."));
        return;
    }

    InferenceActor->RunYOLOInference();
}

void AScenarioManager::FinishScenario()
{
    if (ScenarioState != EScenarioState::Running)
    {
        return;
    }

    GetWorldTimerManager().ClearTimer(InferenceTimerHandle);
    ClearPedestrianStartTimers();

    for (AActor* Pedestrian : Pedestrians)
    {
        if (IsValid(Pedestrian))
        {
            StopPedestrianActor(Pedestrian);
        }
    }

    if (IsValid(Forklift))
    {
        if (UForkliftRouteFollower* RouteFollower = Forklift->FindComponentByClass<UForkliftRouteFollower>())
        {
            RouteFollower->StopFollowing();
        }
    }

    ScenarioState = EScenarioState::Completed;

    if (IsValid(InferenceActor))
    {
        InferenceActor->EndEvaluationRun(BuildRunMetadataFields());
    }

    UE_LOG(LogTemp, Log, TEXT("ScenarioManager: Scenario completed: %s rep %d"), *GetScenarioIdString(), CurrentRepetition);

    // Batch: schedule the next run.
    if (bBatchActive)
    {
        AdvanceBatch();
    }
}

// ============================================================================================
// Helpers
// ============================================================================================

void AScenarioManager::ClearPedestrianStartTimers()
{
    PendingPedestrianStarts.Reset();

    for (FTimerHandle& Handle : PedestrianStartTimers)
    {
        GetWorldTimerManager().ClearTimer(Handle);
    }
    PedestrianStartTimers.Reset();
}

void AScenarioManager::ApplyRepetitionVariation()
{
    if (!bApplyVariationThisRun)
    {
        return;
    }

    FRandomStream Stream(CurrentSeed);

    for (FScenarioPedestrianConfig& Config : ResolvedPedestrianConfigs)
    {
        if (Config.bIsStationary)
        {
            continue;
        }

        Config.MovementSpeed *= 1.0f + Stream.FRandRange(-PedestrianSpeedJitterFraction, PedestrianSpeedJitterFraction);
        const float TimingJitter = Stream.FRandRange(-StartDelayJitterSeconds, StartDelayJitterSeconds);
        Config.StartDelaySeconds = FMath::Max(0.0f, Config.StartDelaySeconds + TimingJitter);
        Config.ArrivalOffsetS += TimingJitter;   // ArrivalOffset mode: same jitter, in seconds
    }
}

float AScenarioManager::GetForkliftForwardSpeedKmh() const
{
    if (!IsValid(Forklift))
    {
        return 0.0f;
    }

    const double ForwardCmPerS = FVector::DotProduct(Forklift->GetVelocity(), Forklift->GetActorForwardVector());
    return static_cast<float>(ForwardCmPerS * 0.036);
}

FString AScenarioManager::GetScenarioIdString() const
{
    if (!IsValid(ScenarioConfig))
    {
        return TEXT("None");
    }

    const FString BaseId = ScenarioConfig->ScenarioId.IsNone() ? ScenarioConfig->GetName() : ScenarioConfig->ScenarioId.ToString();
    return BaseId + CurrentVariantSuffix;
}

FString AScenarioManager::BuildRunMetadataFields() const
{
    const UEnum* TypeEnum = StaticEnum<EScenarioType>();
    const UEnum* ReasonEnum = StaticEnum<EScenarioEndReason>();

    FString Fields = FString::Printf(
        TEXT("\"batch_id\": \"%s\",\n")
        TEXT("  \"scenario_type\": \"%s\",\n")
        TEXT("  \"end_reason\": \"%s\",\n")
        TEXT("  \"seed\": %d,\n")
        TEXT("  \"randomised\": %s,\n")
        TEXT("  \"batch_run_number\": %d,\n")
        TEXT("  \"batch_run_count\": %d,\n")
        TEXT("  \"speed_jitter_fraction\": %.3f,\n")
        TEXT("  \"start_jitter_s\": %.3f,\n")
        TEXT("  \"variant_suffix\": \"%s\",\n")
        TEXT("  \"timing_shift_s\": %.2f,\n")
        TEXT("  \"forklift_target_kmh\": %.2f,\n")
        TEXT("  \"forklift_speed_at_capture_kmh\": %.2f,\n")
        TEXT("  \"waited_for_speed_s\": %.2f,\n")
        TEXT("  \"speed_wait_timed_out\": %s,\n")
        TEXT("  \"inference_interval_s\": %.3f,\n")
        TEXT("  \"duration_s\": %.2f,\n")
        TEXT("  \"captured_s\": %.2f,\n")
        TEXT("  \"use_occluder\": %s,\n")
        TEXT("  \"occluder_actors\": \"%s\",\n")
        TEXT("  \"occluder_tag\": \"%s\",\n")
        TEXT("  \"pedestrians\": ["),
        *CurrentBatchId,
        TypeEnum ? *TypeEnum->GetNameStringByValue(static_cast<int64>(ScenarioConfig->ScenarioType)) : TEXT(""),
        ReasonEnum ? *ReasonEnum->GetNameStringByValue(static_cast<int64>(ScenarioEndReason)) : TEXT(""),
        CurrentSeed,
        bApplyVariationThisRun ? TEXT("true") : TEXT("false"),
        bApplyVariationThisRun ? CurrentBatchRunNumber : 0,
        bApplyVariationThisRun ? BatchRunTotal : 0,
        bApplyVariationThisRun ? PedestrianSpeedJitterFraction : 0.0f,
        bApplyVariationThisRun ? StartDelayJitterSeconds : 0.0f,
        *CurrentVariantSuffix,
        CurrentTimingShiftS,
        CurrentSpeedKmh,
        ForkliftSpeedAtCaptureKmh,
        WaitForSpeedElapsed,
        bSpeedWaitTimedOut ? TEXT("true") : TEXT("false"),
        InferenceIntervalSeconds,
        ScenarioConfig->DurationSeconds,
        ScenarioElapsedTime,
        ScenarioConfig->bUseOccluder ? TEXT("true") : TEXT("false"),
        *ActiveOccluderNames,
        (!ScenarioConfig->bUseOccluder || ScenarioConfig->OccluderActorTag.IsNone()) ? TEXT("") : *ScenarioConfig->OccluderActorTag.ToString());

    const int32 Count = FMath::Min(Pedestrians.Num(), ResolvedPedestrianConfigs.Num());
    for (int32 i = 0; i < Count; ++i)
    {
        const FScenarioPedestrianConfig& Config = ResolvedPedestrianConfigs[i];
        static const TCHAR* ModeNames[] = { TEXT("delay"), TEXT("forklift_distance"), TEXT("arrival_offset") };
        FVector ConflictPoint;
        float WalkToConflictCm = 0.0f;
        FindConflictOnPath(Config, ConflictPoint, WalkToConflictCm);
        const int32 ModeIdx = FMath::Clamp(static_cast<int32>(Config.StartMode), 0, 2);
        Fields += FString::Printf(
            TEXT("%s\n    { \"index\": %d, \"actor_name\": \"%s\", \"speed_cm_s\": %.1f, \"stationary\": %s, \"crouched\": %s, \"start_mode\": \"%s\", \"start_delay_s\": %.3f, \"arrival_offset_s\": %.3f, \"conflict_waypoint_index\": %d, \"conflict_x_m\": %.3f, \"conflict_y_m\": %.3f, \"walk_to_conflict_m\": %.3f, \"trigger_distance_m\": %.3f, \"started_at_s\": %.3f, \"forklift_distance_at_start_m\": %.3f }"),
            i == 0 ? TEXT("") : TEXT(","),
            i,
            IsValid(Pedestrians[i]) ? *Pedestrians[i]->GetName() : TEXT(""),
            Config.MovementSpeed,
            Config.bIsStationary ? TEXT("true") : TEXT("false"),
            Config.bCrouched ? TEXT("true") : TEXT("false"),
            ModeNames[ModeIdx],
            Config.StartDelaySeconds,
            Config.ArrivalOffsetS + CurrentTimingShiftS,
            Config.ConflictWaypointIndex,
            ConflictPoint.X / 100.0,
            ConflictPoint.Y / 100.0,
            WalkToConflictCm / 100.0f,
            PedestrianTriggerDistanceM.IsValidIndex(i) ? PedestrianTriggerDistanceM[i] : -1.0f,
            PedestrianStartTimeS.IsValidIndex(i) ? PedestrianStartTimeS[i] : -1.0f,
            PedestrianForkliftDistanceAtStartM.IsValidIndex(i) ? PedestrianForkliftDistanceAtStartM[i] : -1.0f);
    }
    Fields += TEXT("\n  ]");

    if (IsValid(Forklift))
    {
        if (const USafetyBrakeController* Safety = Forklift->FindComponentByClass<USafetyBrakeController>())
        {
            Fields += TEXT(",\n  ") + Safety->BuildMetadataFields();
        }
    }

    return Fields;
}

// ============================================================================================
// Distance-triggered pedestrian starts
// ============================================================================================

float AScenarioManager::GetFrontOffsetM() const
{
    if (IsValid(Forklift))
    {
        if (const USafetyBrakeController* Safety = Forklift->FindComponentByClass<USafetyBrakeController>())
        {
            return Safety->FrontOffsetFromOriginM;
        }
    }
    return FallbackFrontOffsetM;
}

bool AScenarioManager::FindConflictOnPath(const FScenarioPedestrianConfig& Config, FVector& OutPoint, float& OutWalkCm) const
{
    OutPoint = Config.StartLocation;
    OutWalkCm = 0.0f;

    // Explicit waypoint: walk along the path up to that waypoint.
    if (Config.ConflictWaypointIndex >= 0 && Config.Waypoints.IsValidIndex(Config.ConflictWaypointIndex))
    {
        FVector Previous = Config.StartLocation;
        for (int32 i = 0; i <= Config.ConflictWaypointIndex; ++i)
        {
            OutWalkCm += FVector::Dist2D(Previous, Config.Waypoints[i]);
            Previous = Config.Waypoints[i];
        }
        OutPoint = Config.Waypoints[Config.ConflictWaypointIndex];
        return true;
    }

    // Automatic (-1): the point of the pedestrian's path closest to the forklift route.
    if (Config.Waypoints.Num() == 0 || !IsValid(ForkliftRoute) || !ForkliftRoute->GetSpline())
    {
        return true; // standing pedestrian: its own location
    }

    const USplineComponent* Spline = ForkliftRoute->GetSpline();
    float BestDistSq = TNumericLimits<float>::Max();
    float WalkSoFarCm = 0.0f;
    FVector SegmentStart = Config.StartLocation;

    for (const FVector& SegmentEnd : Config.Waypoints)
    {
        const float SegmentLengthCm = FVector::Dist2D(SegmentStart, SegmentEnd);
        const int32 Steps = FMath::Max(1, FMath::CeilToInt(SegmentLengthCm / 10.0f)); // sample every ~10 cm

        for (int32 Step = 0; Step <= Steps; ++Step)
        {
            const float Alpha = static_cast<float>(Step) / Steps;
            const FVector Sample = FMath::Lerp(SegmentStart, SegmentEnd, Alpha);
            const FVector OnSpline = Spline->FindLocationClosestToWorldLocation(Sample, ESplineCoordinateSpace::World);
            const float DistSq = static_cast<float>(FVector::DistSquared2D(Sample, OnSpline));

            if (DistSq < BestDistSq)
            {
                BestDistSq = DistSq;
                OutPoint = Sample;
                OutWalkCm = WalkSoFarCm + Alpha * SegmentLengthCm;
            }
        }

        WalkSoFarCm += SegmentLengthCm;
        SegmentStart = SegmentEnd;
    }

    const float LateralM = FMath::Sqrt(BestDistSq) / 100.0f;
    if (LateralM > 1.5f)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("ScenarioManager: pedestrian path never comes closer than %.2f m to the forklift route; conflict point may be wrong."),
            LateralM);
    }
    return true;
}

FVector AScenarioManager::GetConflictLocation(const FScenarioPedestrianConfig& Config) const
{
    FVector Point;
    float WalkCm = 0.0f;
    FindConflictOnPath(Config, Point, WalkCm);
    return Point;
}

float AScenarioManager::GetForkliftFrontDistanceAlongRouteM(const FVector& Point) const
{
    if (!IsValid(Forklift) || !IsValid(ForkliftRoute) || !ForkliftRoute->GetSpline())
    {
        return TNumericLimits<float>::Max();
    }

    const USplineComponent* Spline = ForkliftRoute->GetSpline();

    const FVector ForkliftOnSpline = Spline->FindLocationClosestToWorldLocation(Forklift->GetActorLocation(), ESplineCoordinateSpace::World);
    const FVector PointOnSpline = Spline->FindLocationClosestToWorldLocation(Point, ESplineCoordinateSpace::World);

    const float ForkliftAlongCm = Spline->GetDistanceAlongSplineAtLocation(ForkliftOnSpline, ESplineCoordinateSpace::World);
    const float PointAlongCm = Spline->GetDistanceAlongSplineAtLocation(PointOnSpline, ESplineCoordinateSpace::World);

    return (PointAlongCm - ForkliftAlongCm) / 100.0f - GetFrontOffsetM();
}

float AScenarioManager::ComputeTriggerDistanceM(const FScenarioPedestrianConfig& Config) const
{
    if (Config.StartMode == EPedestrianStartMode::ForkliftDistance)
    {
        return Config.StartDistanceM;
    }

    // ArrivalOffset: walking time from the start to the conflict point.
    FVector ConflictPoint;
    float WalkCm = 0.0f;
    FindConflictOnPath(Config, ConflictPoint, WalkCm);

    const float WalkTimeS = (Config.MovementSpeed > KINDA_SMALL_NUMBER) ? WalkCm / Config.MovementSpeed : 0.0f;
    const float ForkliftMps = CurrentSpeedKmh / 3.6f;
    const float OffsetS = Config.ArrivalOffsetS + CurrentTimingShiftS;

    // Start when the forklift is (walk time + offset) seconds away from the conflict point.
    return ForkliftMps * (WalkTimeS + OffsetS);
}

void AScenarioManager::UpdatePendingPedestrianStarts()
{
    for (int32 i = PendingPedestrianStarts.Num() - 1; i >= 0; --i)
    {
        const FPendingPedestrianStart& Pending = PendingPedestrianStarts[i];
        const float DistanceM = GetForkliftFrontDistanceAlongRouteM(Pending.ConflictLocation);

        if (DistanceM > Pending.TriggerDistanceM)
        {
            continue;
        }

        const int32 Index = Pending.PedestrianIndex;
        if (Pedestrians.IsValidIndex(Index) && IsValid(Pedestrians[Index]) && ResolvedPedestrianConfigs.IsValidIndex(Index))
        {
            StartPedestrianActor(Pedestrians[Index], ResolvedPedestrianConfigs[Index]);

            if (PedestrianStartTimeS.IsValidIndex(Index))
            {
                PedestrianStartTimeS[Index] = ScenarioElapsedTime;
                PedestrianForkliftDistanceAtStartM[Index] = DistanceM;
            }

            UE_LOG(LogTemp, Log, TEXT("ScenarioManager: pedestrian %d started (forklift front %.2f m from conflict point, trigger %.2f m)."),
                Index, DistanceM, Pending.TriggerDistanceM);
        }

        PendingPedestrianStarts.RemoveAt(i);
    }
}

void AScenarioManager::ResetPedestrianActor(AActor* Pedestrian, const FScenarioPedestrianConfig& Config)
{
    APedestrianCharacter* PedestChar = Cast<APedestrianCharacter>(Pedestrian);

    if (!PedestChar)
    {
        UE_LOG(
            LogTemp,
            Warning,
            TEXT("ScenarioManager: Actor is not an APedestrianCharacter.")
        );
        return;
    }

    const FTransform StartTransform(
        Config.StartRotation,
        Config.StartLocation
    );

    PedestChar->ScenarioReset(StartTransform);
    PedestChar->ScenarioSetCrouched(Config.bCrouched);
}

void AScenarioManager::StartPedestrianActor(AActor* Pedestrian, const FScenarioPedestrianConfig& Config)
{
    APedestrianCharacter* PedestChar = Cast<APedestrianCharacter>(Pedestrian);

    if (!PedestChar)
    {
        UE_LOG(
            LogTemp,
            Warning,
            TEXT("ScenarioManager: Actor is not an APedestrianCharacter.")
        );
        return;
    }

    if (Config.bIsStationary)
    {
        return;
    }

    PedestChar->ScenarioStart(
        Config.Waypoints,
        Config.MovementSpeed
    );
}

void AScenarioManager::StopPedestrianActor(AActor* Pedestrian)
{
    APedestrianCharacter* PedestChar = Cast<APedestrianCharacter>(Pedestrian);

    if (!PedestChar)
    {
        return;
    }

    PedestChar->ScenarioStop();
}

void AScenarioManager::ActivatePedestrian(AActor* Pedestrian)
{
    Pedestrian->SetActorHiddenInGame(false);
    Pedestrian->SetActorEnableCollision(true);

    if (ACharacter* Character = Cast<ACharacter>(Pedestrian))
    {
        if (UCharacterMovementComponent* Movement = Character->GetCharacterMovement())
        {
            if (Movement->MovementMode == MOVE_None)
            {
                Movement->SetMovementMode(MOVE_Walking);
            }
        }
    }
}

void AScenarioManager::ParkUnusedPedestrians(int32 FirstUnusedIndex)
{
    for (int32 Index = FirstUnusedIndex; Index < Pedestrians.Num(); ++Index)
    {
        AActor* Pedestrian = Pedestrians[Index];
        if (!IsValid(Pedestrian))
        {
            continue;
        }

        StopPedestrianActor(Pedestrian);

        if (ACharacter* Character = Cast<ACharacter>(Pedestrian))
        {
            if (UCharacterMovementComponent* Movement = Character->GetCharacterMovement())
            {
                Movement->StopMovementImmediately();
                Movement->DisableMovement();   // MOVE_None: no gravity, so it cannot fall once collision is off
            }
        }

        if (InitialPedestrianTransforms.IsValidIndex(Index))
        {
            Pedestrian->SetActorTransform(InitialPedestrianTransforms[Index], false, nullptr, ETeleportType::TeleportPhysics);
        }

        Pedestrian->SetActorEnableCollision(false);
        Pedestrian->SetActorHiddenInGame(true);

        UE_LOG(LogTemp, Log, TEXT("ScenarioManager: pedestrian %d (%s) not used by this scenario: parked and hidden."),
            Index, *Pedestrian->GetName());
    }
}

ATargetPoint* AScenarioManager::FindScenarioPointByTag(FName PointId) const
{
    if (PointId.IsNone() || !GetWorld())
    {
        return nullptr;
    }

    ATargetPoint* FoundPoint = nullptr;

    for (TActorIterator<ATargetPoint> It(GetWorld()); It; ++It)
    {
        ATargetPoint* Point = *It;

        if (!IsValid(Point) || !Point->ActorHasTag(PointId))
        {
            continue;
        }

        if (FoundPoint)
        {
            UE_LOG(
                LogTemp,
                Error,
                TEXT(
                    "ScenarioManager: Duplicate TargetPoint tag '%s'."
                ),
                *PointId.ToString()
            );

            return nullptr;
        }

        FoundPoint = Point;
    }

    return FoundPoint;
}

bool AScenarioManager::ResolveScenarioPoint(FName PointId, FVector& OutLocation) const
{
    ATargetPoint* Point =
        FindScenarioPointByTag(PointId);

    if (!Point)
    {
        UE_LOG(
            LogTemp,
            Error,
            TEXT(
                "ScenarioManager: TargetPoint with tag '%s' "
                "was not found."
            ),
            *PointId.ToString()
        );

        return false;
    }

    OutLocation = Point->GetActorLocation();
    return true;
}

bool AScenarioManager::ResolveScenarioPointTransform(
    FName PointId,
    FTransform& OutTransform
) const
{
    ATargetPoint* Point =
        FindScenarioPointByTag(PointId);

    if (!Point)
    {
        UE_LOG(
            LogTemp,
            Error,
            TEXT(
                "ScenarioManager: TargetPoint with tag '%s' "
                "was not found."
            ),
            *PointId.ToString()
        );

        return false;
    }

    OutTransform = Point->GetActorTransform();

    return true;
}

bool AScenarioManager::ResolvePedestrianConfig(
    const FScenarioPedestrianConfig& SourceConfig,
    FScenarioPedestrianConfig& OutConfig
) const
{
    OutConfig = SourceConfig;

    if (!SourceConfig.StartPointId.IsNone())
    {
        FTransform StartTransform;

        if (!ResolveScenarioPointTransform(
            SourceConfig.StartPointId,
            StartTransform
        ))
        {
            return false;
        }

        OutConfig.StartLocation =
            StartTransform.GetLocation();

        OutConfig.StartRotation =
            StartTransform.GetRotation().Rotator();

        // Keep the pedestrian upright.
        OutConfig.StartRotation.Pitch = 0.0f;
        OutConfig.StartRotation.Roll = 0.0f;
    }

    if (SourceConfig.WaypointIds.Num() > 0)
    {
        OutConfig.Waypoints.Empty();

        for (const FName& PointId :
            SourceConfig.WaypointIds)
        {
            FVector Location;

            if (!ResolveScenarioPoint(PointId, Location))
            {
                return false;
            }

            OutConfig.Waypoints.Add(Location);
        }
    }

    return true;
}

bool AScenarioManager::ResolveScenarioConfiguration()
{
    if (!IsValid(ScenarioConfig))
    {
        return false;
    }

    ResolvedPedestrianConfigs.Empty();
    ResolvedPedestrianConfigs.Reserve(
        ScenarioConfig->Pedestrians.Num()
    );

    for (const FScenarioPedestrianConfig& SourceConfig :
        ScenarioConfig->Pedestrians)
    {
        FScenarioPedestrianConfig ResolvedConfig;

        if (!ResolvePedestrianConfig(
            SourceConfig,
            ResolvedConfig
        ))
        {
            return false;
        }

        UE_LOG(
            LogTemp,
            Log,
            TEXT(
                "Pedestrian resolved: Start=%s, Waypoints=%d"
            ),
            *ResolvedConfig.StartLocation.ToString(),
            ResolvedConfig.Waypoints.Num()
        );

        ResolvedPedestrianConfigs.Add(ResolvedConfig);
    }

    return true;
}

AForkliftRoute* AScenarioManager::FindForkliftRouteByTag(
    FName RouteId
) const
{
    if (RouteId.IsNone() || !GetWorld())
    {
        return nullptr;
    }

    AForkliftRoute* FoundRoute = nullptr;

    for (
        TActorIterator<AForkliftRoute> It(GetWorld());
        It;
        ++It
        )
    {
        AForkliftRoute* Route = *It;

        if (!IsValid(Route) ||
            !Route->ActorHasTag(RouteId))
        {
            continue;
        }

        if (FoundRoute)
        {
            UE_LOG(
                LogTemp,
                Error,
                TEXT(
                    "ScenarioManager: Duplicate ForkliftRoute "
                    "tag '%s'."
                ),
                *RouteId.ToString()
            );

            return nullptr;
        }

        FoundRoute = Route;
    }

    return FoundRoute;
}

bool AScenarioManager::ResolveForkliftRoute()
{
    ForkliftRoute = nullptr;

    if (!IsValid(ScenarioConfig))
    {
        return false;
    }

    if (ScenarioConfig->ForkliftRouteId.IsNone())
    {
        return true;
    }

    AForkliftRoute* FoundRoute =
        FindForkliftRouteByTag(
            ScenarioConfig->ForkliftRouteId
        );

    if (!FoundRoute)
    {
        UE_LOG(
            LogTemp,
            Warning,
            TEXT(
                "ScenarioManager: Forklift route '%s' "
                "could not be resolved. Falling back."
            ),
            *ScenarioConfig->ForkliftRouteId.ToString()
        );

        return true;
    }

    ForkliftRoute = FoundRoute;

    UE_LOG(
        LogTemp,
        Log,
        TEXT(
            "Forklift route resolved: ID='%s', Actor='%s'"
        ),
        *ScenarioConfig->ForkliftRouteId.ToString(),
        *FoundRoute->GetName()
    );

    return true;
}

bool AScenarioManager::ResolveForkliftStartTransform()
{
    if (!IsValid(ScenarioConfig))
    {
        return false;
    }

    // Priority 1:
    // Selected route's first spline point.
    if (IsValid(ForkliftRoute))
    {
        ResolvedForkliftStartLocation =
            ForkliftRoute->GetStartLocation();

        ResolvedForkliftStartRotation =
            ForkliftRoute->GetStartRotation();

        UE_LOG(
            LogTemp,
            Log,
            TEXT(
                "Forklift resolved from route '%s': "
                "Location=%s Rotation=%s"
            ),
            *ScenarioConfig->ForkliftRouteId.ToString(),
            *ResolvedForkliftStartLocation.ToString(),
            *ResolvedForkliftStartRotation.ToString()
        );

        return true;
    }

    // Priority 2:
    // Explicit TargetPoint location.
    if (!ScenarioConfig->ForkliftStartPointId.IsNone())
    {
        if (ResolveScenarioPoint(
            ScenarioConfig->ForkliftStartPointId,
            ResolvedForkliftStartLocation
        ))
        {
            // TargetPoint rotation is deliberately ignored.
            ResolvedForkliftStartRotation =
                ScenarioConfig->ForkliftStartRotation;

            UE_LOG(
                LogTemp,
                Log,
                TEXT(
                    "Forklift resolved from TargetPoint '%s': "
                    "Location=%s Rotation=%s"
                ),
                *ScenarioConfig->ForkliftStartPointId.ToString(),
                *ResolvedForkliftStartLocation.ToString(),
                *ResolvedForkliftStartRotation.ToString()
            );

            return true;
        }

        UE_LOG(
            LogTemp,
            Warning,
            TEXT(
                "ScenarioManager: Forklift start point '%s' "
                "could not be resolved. Falling back."
            ),
            *ScenarioConfig->ForkliftStartPointId.ToString()
        );
    }

    // Priority 3:
    // Manual Data Asset values.
    ResolvedForkliftStartLocation =
        ScenarioConfig->ForkliftStartLocation;

    ResolvedForkliftStartRotation =
        ScenarioConfig->ForkliftStartRotation;

    UE_LOG(
        LogTemp,
        Log,
        TEXT(
            "Forklift resolved from manual fallback: "
            "Location=%s Rotation=%s"
        ),
        *ResolvedForkliftStartLocation.ToString(),
        *ResolvedForkliftStartRotation.ToString()
    );

    return true;
}

bool AScenarioManager::HasForkliftReachedRouteEnd() const
{
    if (!IsValid(Forklift) ||
        !IsValid(ForkliftRoute))
    {
        return false;
    }

    UForkliftRouteFollower* RouteFollower =
        Forklift->FindComponentByClass<
        UForkliftRouteFollower>();

    if (!RouteFollower)
    {
        return false;
    }

    return RouteFollower->HasReachedEnd();
}

bool AScenarioManager::IsForkliftStopped() const
{
    const float StopSpeedCmPerSecond = 20.0f;

    if (!IsValid(Forklift))
    {
        return false;
    }

    const FVector Velocity =
        Forklift->GetVelocity();

    return Velocity.SizeSquared() <=
        FMath::Square(StopSpeedCmPerSecond);
}