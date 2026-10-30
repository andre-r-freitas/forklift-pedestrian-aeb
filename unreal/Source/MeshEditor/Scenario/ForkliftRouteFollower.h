#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ForkliftRouteFollower.generated.h"

class USplineComponent;

USTRUCT()
struct FPIDSpring
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, Category = "PID")
    float Kp = 1.0f;

    UPROPERTY(EditAnywhere, Category = "PID")
    float Ki = 0.1f;

    UPROPERTY(EditAnywhere, Category = "PID")
    float Kd = 0.2f;

    /** Anti-windup: the integral term may contribute at most this much output (e.g. 0.15 = 15% throttle). */
    UPROPERTY(EditAnywhere, Category = "PID")
    float MaxIntegralContribution = 0.6f;

    float Integral = 0.0f;
    float PrevError = 0.0f;

    float Update(float Error, float DeltaTime, float MinOutput = 0.0f, float MaxOutput = 1.0f)
    {
        if (DeltaTime <= KINDA_SMALL_NUMBER)
        {
            return 0.0f;
        }

        // (The old "reset the integral when the error changes sign" rule is gone: the ramp-up is now done by the
        // launch phase, so no integral builds up during acceleration, and the reset caused a speed drop and a
        // steady-state error below the target.)

        const float Derivative = (Error - PrevError) / DeltaTime;
        const float Unclamped = Kp * Error + Ki * (Integral + Error * DeltaTime) + Kd * Derivative;

        // Anti-windup 2: only integrate while the output is not saturated in the direction of the error.
        const bool bSaturatedHigh = Unclamped >= MaxOutput && Error > 0.0f;
        const bool bSaturatedLow = Unclamped <= MinOutput && Error < 0.0f;
        if (!bSaturatedHigh && !bSaturatedLow)
        {
            Integral += Error * DeltaTime;
        }

        // Anti-windup 3: cap the integral contribution.
        if (Ki > KINDA_SMALL_NUMBER)
        {
            const float Limit = MaxIntegralContribution / Ki;
            Integral = FMath::Clamp(Integral, -Limit, Limit);
        }

        float Output = Kp * Error + Ki * Integral + Kd * Derivative;
        Output = FMath::Clamp(Output, MinOutput, MaxOutput);

        PrevError = Error;
        return Output;
    }

    void Reset()
    {
        Integral = 0.0f;
        PrevError = 0.0f;
    }

    /** Clears the integral and sets the previous error, so the first Update has no derivative kick. */
    void Prime(float Error)
    {
        Integral = 0.0f;
        PrevError = Error;
    }
};

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class MESHEDITOR_API UForkliftRouteFollower : public UActorComponent
{
    GENERATED_BODY()

public:
    // Sets default values for this component's properties
    UForkliftRouteFollower();

    // Called every frame
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION(BlueprintCallable, Category = "Route")
    void StartFollowing(USplineComponent* InSpline, float InSpeedKmh);

    UFUNCTION(BlueprintCallable, Category = "Route")
    void StopFollowing();

    UFUNCTION(BlueprintCallable, Category = "Route")
    void ResetFollower();

    UFUNCTION(BlueprintCallable, Category = "Route")
    bool HasReachedEnd() const;

    /**
     * Closed loop: stop following and brake with the given input (0-1) until stopped, then hold
     * with the handbrake (avoids Chaos "reverse as brake" once the vehicle is at standstill).
     * Cleared by ResetFollower / StartFollowing.
     */
    UFUNCTION(BlueprintCallable, Category = "Route")
    void EmergencyStop(float BrakeInput);

    UFUNCTION(BlueprintPure, Category = "Route")
    bool IsEmergencyStopping() const { return bEmergencyStop; }

    // Tunable parameters
    UPROPERTY(EditAnywhere, Category = "Tuning")
    float LookaheadDistanceCm = 500.0f;

    UPROPERTY(EditAnywhere, Category = "Tuning")
    float SteeringGain = 1.2f;

    UPROPERTY(EditAnywhere, Category = "Tuning")
    float MinSpeedKmh = 2.0f;

    UPROPERTY(EditAnywhere, Category = "Tuning")
    float MaxSpeedKmh = 20.0f;

    /**
     * Launch phase: below LaunchUntilFraction x target speed the forklift accelerates with LaunchThrottle
     * (no PID), then the PID takes over to hold the speed. Faster, more repeatable start.
     */
    UPROPERTY(EditAnywhere, Category = "Tuning", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float LaunchThrottle = 1.0f;

    UPROPERTY(EditAnywhere, Category = "Tuning", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float LaunchUntilFraction = 0.9f;

    UPROPERTY(EditAnywhere, Category = "PID")
    FPIDSpring SpeedPID;

protected:
    // Called when the game starts
    virtual void BeginPlay() override;

    void UpdateDriving(float DeltaTime);
    float GetForwardSpeedCmPerSecond() const;
    class UChaosWheeledVehicleMovementComponent* GetVehicleMovement() const;

    TObjectPtr<USplineComponent> Spline;
    float SpeedKmh;
    bool bFollowing;
    bool bReachedEnd;
    bool bEmergencyStop = false;
    float EmergencyBrakeInput = 0.0f;
};
