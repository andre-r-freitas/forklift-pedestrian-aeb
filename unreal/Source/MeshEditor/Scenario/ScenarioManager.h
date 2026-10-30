#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ScenarioConfigDataAsset.h"
#include "ScenarioManager.generated.h"

class AYOLOInferenceActor;
class APedestrianCharacter;
class AForkliftRoute;
class ATargetPoint;

UENUM(BlueprintType)
enum class EScenarioState : uint8
{
    Idle            UMETA(DisplayName = "Idle"),
    Resetting       UMETA(DisplayName = "Resetting"),
    Ready           UMETA(DisplayName = "Ready"),
    WarmingUp       UMETA(DisplayName = "Warming Up"),
    WaitingForSpeed UMETA(DisplayName = "Waiting For Speed"),
    Running         UMETA(DisplayName = "Running"),
    Completed       UMETA(DisplayName = "Completed"),
    Failed          UMETA(DisplayName = "Failed")
};

UENUM(BlueprintType)
enum class EScenarioEndReason : uint8
{
    None,
    RouteCompleted,
    DurationExpired,
    Stopped,
    SafetyStop,   // closed loop: the forklift stopped for a person and was held for SafetyStopEndDelayS
    Aborted       // play ended (editor stopped) while the run was active
};

/**
 * Run flow per scenario (single run or one batch repetition):
 *   Reset -> WarmingUp (actors settle, WarmupTimeSeconds)
 *         -> WaitingForSpeed (forklift starts driving; capture begins once it reaches
 *            AtSpeedFraction x target speed, or after MaxWaitForSpeedSeconds)
 *         -> Running (pedestrians start, inference + logging) -> Completed
 *
 * NOTE: the forklift now travels a short distance (about 0.3 m at 5 km/h, 2.6 m at 15 km/h
 * with the current vehicle tuning) before capture and pedestrians start. Check that
 * pedestrian start delays still create the intended conflicts.
 */
 /** One run of a batch: scenario asset + variant (speed, timing shift) + repetition. */
struct FBatchRunSpec
{
    int32 ScenarioIndex = 0;
    int32 VariantIndex = 0;      // unique per (scenario, speed, timing shift); NOT used for the seed (see StartCurrentBatchRun)
    float SpeedKmh = 0.0f;
    float TimingShiftS = 0.0f;
    FString Suffix;              // e.g. "_v10_t+1.0"
    int32 Repetition = 0;
    int32 RunNumber = 0;         // 1-based position in the full batch list (ResumeFromRun / OnlyRunNumbers)
};

/** A pedestrian waiting to be started by forklift distance. */
struct FPendingPedestrianStart
{
    int32 PedestrianIndex = 0;
    float TriggerDistanceM = 0.0f;
    FVector ConflictLocation = FVector::ZeroVector;
};

UCLASS()
class MESHEDITOR_API AScenarioManager : public AActor
{
    GENERATED_BODY()

public:
    AScenarioManager();

    /** Runs ScenarioConfig once (no randomisation). */
    UFUNCTION(BlueprintCallable, Category = "Scenario")
    void StartScenario();

    UFUNCTION(BlueprintCallable, Category = "Scenario")
    void ResetScenario();

    /** Runs every scenario in BatchScenarios RepetitionsPerScenario times, one after the other. */
    UFUNCTION(BlueprintCallable, Category = "Scenario|Batch")
    void StartBatch();

    UFUNCTION(BlueprintCallable, Category = "Scenario|Batch")
    void StopBatch();

protected:
    virtual void BeginPlay() override;

    virtual void Tick(float DeltaSeconds) override;

    UFUNCTION()
    void OnWarmupFinished();

    UFUNCTION()
    void BeginScenarioCapture();

    UFUNCTION()
    void RunScenarioInference();

    UFUNCTION()
    void FinishScenario();

    UFUNCTION()
    void ResetPedestrianActor(AActor* Pedestrian, const FScenarioPedestrianConfig& Config);

    UFUNCTION()
    void StartPedestrianActor(AActor* Pedestrian, const FScenarioPedestrianConfig& Config);

    UFUNCTION()
    void StopPedestrianActor(AActor* Pedestrian);

    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario")
    TObjectPtr<UScenarioConfigDataAsset> ScenarioConfig;

    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Actors")
    TObjectPtr<AActor> Forklift;

    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Actors")
    TArray<TObjectPtr<AActor>> Pedestrians;

    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Actors")
    TObjectPtr<AActor> OccluderActor;

    /**
     * Further occluders (e.g. a 1.2 m and a 1.8 m fork load). Together with OccluderActor these are all hidden at reset,
     * except the ones the scenario asks for (bUseOccluder + OccluderActorTag on the Data Asset).
     */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Actors")
    TArray<TObjectPtr<AActor>> OccluderActors;

    UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "Scenario")
    EScenarioState ScenarioState = EScenarioState::Idle;

    /** Names of the occluder actors shown in the current run (for the metadata). */
    FString ActiveOccluderNames;

    UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "Scenario")
    EScenarioEndReason ScenarioEndReason = EScenarioEndReason::None;

    UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "Scenario")
    float ScenarioElapsedTime = 0.0f;

    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Inference")
    float InferenceIntervalSeconds = 0.20f;

    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Actors")
    TObjectPtr<AYOLOInferenceActor> InferenceActor;

    // ---------------- Forklift start ----------------

    /** Capture starts once the forklift reaches this fraction of its target speed. */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Forklift Start", meta = (ClampMin = "0.1", ClampMax = "1.0"))
    float AtSpeedFraction = 0.95f;

    /** Safety net: capture starts anyway after this long (a warning is logged and stored in the metadata). */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Forklift Start")
    float MaxWaitForSpeedSeconds = 10.0f;

    // ---------------- Closed loop ----------------

    /**
     * Enables automatic emergency braking (USafetyBrakeController on the forklift) for every run started
     * by this manager. Off = detection-quality runs (the forklift never reacts). On = closed-loop runs.
     */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Closed Loop")
    bool bEnableSafetyBraking = false;

    /**
     * After an emergency stop the forklift is held; the run ends this many seconds after it stopped
     * (end reason SafetyStop), so the pedestrian's reaction/clearance after the stop is still logged.
     */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Closed Loop", meta = (ClampMin = "0.0"))
    float SafetyStopEndDelayS = 3.0f;

    // ---------------- Batch ----------------

    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Batch")
    TArray<TObjectPtr<UScenarioConfigDataAsset>> BatchScenarios;

    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Batch", meta = (ClampMin = "1"))
    int32 RepetitionsPerScenario = 10;

    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Batch")
    bool bRunBatchOnBeginPlay = false;

    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Batch")
    float DelayBetweenRunsSeconds = 1.0f;

    /**
     * Resume an interrupted batch: start at this run number (1-based, as in the log line "batch run N/M" and in
     * run_metadata.json "batch_run_number"). The run list, seeds and folder names are rebuilt exactly as in the
     * original batch, so the resumed runs are identical to the ones that did not run. 1 = start from the beginning.
     * Keep BatchScenarios, variants, RepetitionsPerScenario and BaseSeed unchanged when resuming.
     */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Batch", meta = (ClampMin = "1"))
    int32 ResumeFromRun = 1;

    /**
     * Re-run only these run numbers (1-based, same numbering as ResumeFromRun), e.g. runs the evaluator flagged.
     * Empty = all runs (from ResumeFromRun on). Same seeds and folder names as in the original batch.
     */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Batch")
    TArray<int32> OnlyRunNumbers;

    /**
     * Seed per run = BaseSeed + 1000 * scenario index (position in BatchScenarios) + repetition.
     * Reproducible, and the same for every speed / timing variant of a scenario (paired comparison).
     * Reordering BatchScenarios changes the seeds; freeze the order before collecting thesis data.
     */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Batch")
    int32 BaseSeed = 1234;

    /** Pedestrian speed is multiplied by a random factor in [1 - x, 1 + x] per repetition. */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Batch", meta = (ClampMin = "0.0", ClampMax = "0.5"))
    float PedestrianSpeedJitterFraction = 0.10f;

    /** Pedestrian start delay is shifted by a random value in [-x, +x] seconds (clamped at 0). */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Batch", meta = (ClampMin = "0.0"))
    float StartDelayJitterSeconds = 0.5f;

    FTimerHandle WarmupTimerHandle;

    FTimerHandle InferenceTimerHandle;

    FTimerHandle NextRunTimerHandle;

    /** Used for distance-triggered pedestrian starts if the forklift has no SafetyBrakeController. */
    UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Scenario|Forklift Start")
    float FallbackFrontOffsetM = 2.58f;

private:
    ATargetPoint* FindScenarioPointByTag(FName PointId) const;

    bool ResolveScenarioPoint(FName PointId, FVector& OutLocation) const;

    bool ResolveScenarioPointTransform(FName PointId, FTransform& OutTransform) const;

    bool ResolvePedestrianConfig(
        const FScenarioPedestrianConfig& SourceConfig,
        FScenarioPedestrianConfig& OutConfig
    ) const;

    bool ResolveScenarioConfiguration();

    void ApplyRepetitionVariation();

    void StartForklift();

    void StartCurrentBatchRun();

    void BuildBatchRuns();

    void AdvanceBatch();

    float GetFrontOffsetM() const;

    FVector GetConflictLocation(const FScenarioPedestrianConfig& Config) const;

    /** Conflict point and walking distance to it (explicit waypoint, or automatic: path point closest to the route). */
    bool FindConflictOnPath(const FScenarioPedestrianConfig& Config, FVector& OutPoint, float& OutWalkCm) const;

    /** Forklift front distance to a point, measured along the route (m). */
    float GetForkliftFrontDistanceAlongRouteM(const FVector& Point) const;

    float ComputeTriggerDistanceM(const FScenarioPedestrianConfig& Config) const;

    void UpdatePendingPedestrianStarts();

    void ClearPedestrianStartTimers();

    float GetForkliftForwardSpeedKmh() const;

    FString GetScenarioIdString() const;

    FString BuildRunMetadataFields() const;

    TArray<FScenarioPedestrianConfig> ResolvedPedestrianConfigs;

    AForkliftRoute* FindForkliftRouteByTag(FName RouteId) const;

    bool ResolveForkliftRoute();

    bool ResolveForkliftStartTransform();

    FVector ResolvedForkliftStartLocation = FVector::ZeroVector;

    FRotator ResolvedForkliftStartRotation = FRotator::ZeroRotator;

    AForkliftRoute* ForkliftRoute = nullptr;

    bool HasForkliftReachedRouteEnd() const;

    bool IsForkliftStopped() const;

    // Run bookkeeping
    TArray<FTimerHandle> PedestrianStartTimers;
    bool bBatchActive = false;
    TArray<FBatchRunSpec> BatchRuns;
    int32 BatchRunIndex = 0;
    float CurrentSpeedKmh = 0.0f;
    float CurrentTimingShiftS = 0.0f;
    FString CurrentVariantSuffix;
    TArray<FPendingPedestrianStart> PendingPedestrianStarts;
    TArray<float> PedestrianTriggerDistanceM;
    TArray<float> PedestrianStartTimeS;
    TArray<float> PedestrianForkliftDistanceAtStartM;
    FString CurrentBatchId;
    int32 CurrentRepetition = 0;
    int32 CurrentBatchRunNumber = 0;
    int32 BatchRunTotal = 0;
    int32 CurrentSeed = 0;
    bool bApplyVariationThisRun = false;
    float WaitForSpeedElapsed = 0.0f;
    float ForkliftSpeedAtCaptureKmh = 0.0f;
    bool bSpeedWaitTimedOut = false;

    // Level transforms of the pedestrian actors at BeginPlay; pedestrians a scenario does not use are parked there.
    TArray<FTransform> InitialPedestrianTransforms;

    void ParkUnusedPedestrians(int32 FirstUnusedIndex);
    void ActivatePedestrian(AActor* Pedestrian);
};