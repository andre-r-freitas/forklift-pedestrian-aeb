#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ForkliftDynamicsTest.generated.h"

class UChaosWheeledVehicleMovementComponent;

UENUM(BlueprintType)
enum class EDynamicsTestPhase : uint8
{
    Idle,
    Settling,
    Accelerating,
    Cruising,
    Braking,
    Done
};

/**
 * Straight-line acceleration + braking test for the forklift.
 *
 * For each target speed (x repetitions):
 *   1. Reset to the start transform, hold the handbrake while it settles.
 *   2. Full throttle until the target speed is reached (time + distance to target).
 *   3. Hold the target speed for CruiseSeconds (simple P-controller).
 *   4. Full brake until stopped (stopping time + distance + deceleration).
 *
 * Output (Saved/VehicleTests/<timestamp>/):
 *   - summary.csv : one row per run
 *   - ticks.csv   : per-tick speed, inputs, gear, RPM, position
 *
 * Place the forklift at the start of a long, straight, flat stretch facing
 * along it (15 km/h + braking needs roughly 30-40 m of free space).
 * Do NOT start a scenario at the same time: this component drives the vehicle itself.
 */
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class MESHEDITOR_API UForkliftDynamicsTest : public UActorComponent
{
    GENERATED_BODY()

public:
    UForkliftDynamicsTest();

    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION(BlueprintCallable, Category = "Dynamics Test")
    void StartTest();

    UPROPERTY(EditAnywhere, Category = "Dynamics Test")
    bool bAutoStartOnBeginPlay = true;

    UPROPERTY(EditAnywhere, Category = "Dynamics Test")
    TArray<float> TargetSpeedsKmh = { 5.0f, 10.0f, 15.0f };

    UPROPERTY(EditAnywhere, Category = "Dynamics Test", meta = (ClampMin = "1"))
    int32 RepetitionsPerSpeed = 3;

    UPROPERTY(EditAnywhere, Category = "Dynamics Test")
    float SettleSeconds = 1.5f;

    UPROPERTY(EditAnywhere, Category = "Dynamics Test")
    float CruiseSeconds = 2.0f;

    /** Throttle added per km/h of speed error while cruising. */
    UPROPERTY(EditAnywhere, Category = "Dynamics Test")
    float CruiseThrottleGain = 0.1f;

    /** Run ends when forward speed drops to or below this (signed, so reversing also ends it). */
    UPROPERTY(EditAnywhere, Category = "Dynamics Test")
    float StoppedSpeedKmh = 0.1f;

    /** Safety net: a phase that takes longer than this ends the run and marks it timed out. */
    UPROPERTY(EditAnywhere, Category = "Dynamics Test")
    float PhaseTimeoutSeconds = 20.0f;

    /**
     * Service-brake input (0-1) used for the stop. Lower it to find the input that gives your sourced
     * forklift deceleration, then use that value as EmergencyBrakeInput on the SafetyBrakeController.
     */
    UPROPERTY(EditAnywhere, Category = "Dynamics Test", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float StopBrakeInput = 1.0f;

    /** Brake with the handbrake instead of the service brake (to compare both). */
    UPROPERTY(EditAnywhere, Category = "Dynamics Test")
    bool bUseHandbrakeForStop = false;

protected:
    virtual void BeginPlay() override;

private:
    void BeginRun();
    void EnterPhase(EDynamicsTestPhase NewPhase);
    void FinishRun(bool bTimedOut);
    void ApplyInputs(float InThrottle, float InBrake, float InHandbrake);
    void AppendTickLine();
    void FlushLogs();

    UChaosWheeledVehicleMovementComponent* GetVehicle() const;
    float GetSpeedKmh() const;
    float GetCurrentTargetKmh() const;
    float Now() const;
    static float DistanceMeters2D(const FVector& A, const FVector& B);
    static const TCHAR* PhaseName(EDynamicsTestPhase InPhase);

    EDynamicsTestPhase Phase = EDynamicsTestPhase::Idle;

    FTransform StartTransform;
    FString OutputFolder;
    FString TickBuffer;
    FString SummaryBuffer;

    int32 RunIndex = 0;
    int32 TotalRuns = 0;

    float RunStartTime = 0.0f;
    float PhaseStartTime = 0.0f;

    FVector AccelStartLocation = FVector::ZeroVector;
    float AccelStartTime = 0.0f;
    float TimeToTargetS = -1.0f;
    float DistToTargetM = -1.0f;

    FVector BrakeStartLocation = FVector::ZeroVector;
    float BrakeStartTime = 0.0f;
    float SpeedAtBrakeKmh = 0.0f;

    // Braking profile: dead time and steady deceleration (between 80% and 20% of the initial speed)
    float BrakeDeadTimeS = -1.0f;
    float TimeAt80PctS = -1.0f;
    float TimeAt20PctS = -1.0f;

    float CurrentThrottle = 0.0f;
    float CurrentBrake = 0.0f;
    float CurrentHandbrake = 0.0f;
};