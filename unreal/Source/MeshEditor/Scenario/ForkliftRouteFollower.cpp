#include "ForkliftRouteFollower.h"
#include "GameFramework/Actor.h"
#include "ChaosWheeledVehicleMovementComponent.h"
#include "Components/SplineComponent.h"

UForkliftRouteFollower::UForkliftRouteFollower()
{
    PrimaryComponentTick.bCanEverTick = true;

    Spline = nullptr;
    SpeedKmh = 0.0f;
    bFollowing = false;
    bReachedEnd = false;

    // Initial PID gains (tunable options)
    // Gains per cm/s of speed error. Kp 0.01: 1 km/h (28 cm/s) error -> 0.28 throttle.
    // Ki 0.004 with MaxIntegralContribution 0.6 removes the steady-state error within a few seconds.
    SpeedPID.Kp = 0.01f;
    SpeedPID.Ki = 0.004f;
    SpeedPID.Kd = 0.0f;
    SpeedPID.MaxIntegralContribution = 0.6f;
}

void UForkliftRouteFollower::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (bEmergencyStop)
    {
        if (UChaosWheeledVehicleMovementComponent* VehicleMove = GetVehicleMovement())
        {
            const bool bStopped = GetForwardSpeedCmPerSecond() <= 1.0f;
            VehicleMove->SetThrottleInput(0.0f);
            VehicleMove->SetSteeringInput(0.0f);
            VehicleMove->SetBrakeInput(bStopped ? 0.0f : EmergencyBrakeInput);
            VehicleMove->SetHandbrakeInput(bStopped);
        }
        return;
    }

    if (!bFollowing || !Spline)
    {
        return;
    }

    UpdateDriving(DeltaTime);
}

// Called when the game starts or when spawned
void UForkliftRouteFollower::BeginPlay()
{
    Super::BeginPlay();

}

void UForkliftRouteFollower::StartFollowing(USplineComponent* InSpline, float InSpeedKmh)
{
    UE_LOG(
        LogTemp,
        Log,
        TEXT("ForkliftRouteFollower: StartFollowing - Spline=%p, SpeedKmh=%.2f"),
        InSpline,
        InSpeedKmh
    );

    Spline = InSpline;
    SpeedKmh = InSpeedKmh;
    bReachedEnd = false;
    bEmergencyStop = false;
    bFollowing = (Spline != nullptr) && (SpeedKmh > 0.0f);

    SpeedPID.Reset();

    // Release handbrake so we can move
    UChaosWheeledVehicleMovementComponent* VehicleMove = GetVehicleMovement();
    if (VehicleMove)
    {
        VehicleMove->SetHandbrakeInput(0.0f);
    }
}

void UForkliftRouteFollower::StopFollowing()
{
    UE_LOG(
        LogTemp,
        Log,
        TEXT("ForkliftRouteFollower: StopFollowing")
    );

    bFollowing = false;

    UChaosWheeledVehicleMovementComponent* VehicleMove = GetVehicleMovement();
    if (VehicleMove)
    {
        VehicleMove->SetThrottleInput(0.0f);
        VehicleMove->SetBrakeInput(0.0f);
        VehicleMove->SetHandbrakeInput(1.0f); // lock in place
        VehicleMove->SetSteeringInput(0.0f);
    }

    SpeedPID.Reset();
}

void UForkliftRouteFollower::ResetFollower()
{
    UE_LOG(
        LogTemp,
        Log,
        TEXT("ForkliftRouteFollower: ResetFollower")
    );

    bFollowing = false;
    bReachedEnd = false;
    Spline = nullptr;
    bEmergencyStop = false;
    SpeedKmh = 0.0f;

    UChaosWheeledVehicleMovementComponent* VehicleMove = GetVehicleMovement();
    if (VehicleMove)
    {
        VehicleMove->SetThrottleInput(0.0f);
        VehicleMove->SetBrakeInput(0.0f);
        VehicleMove->SetHandbrakeInput(1.0f); // lock in place
        VehicleMove->SetSteeringInput(0.0f);
    }

    SpeedPID.Reset();
}

void UForkliftRouteFollower::UpdateDriving(float DeltaTime)
{
    AActor* Owner = GetOwner();
    if (!Owner || !Spline)
    {
        return;
    }

    UChaosWheeledVehicleMovementComponent* VehicleMove = GetVehicleMovement();
    if (!VehicleMove)
    {
        return;
    }

    const FVector VehicleLoc = Owner->GetActorLocation();
    const FTransform VehicleTransform = Owner->GetActorTransform();

    // Find closest point on spline
    const FVector ClosestPoint =
        Spline->FindLocationClosestToWorldLocation(
            VehicleLoc,
            ESplineCoordinateSpace::World
        );

    const float ClosestDistance =
        Spline->GetDistanceAlongSplineAtLocation(
            ClosestPoint,
            ESplineCoordinateSpace::World
        );

    const float SplineLength = Spline->GetSplineLength();

    // Lookahead target
    const float LookaheadDistance = LookaheadDistanceCm;
    float TargetDistance = ClosestDistance + LookaheadDistance;
    TargetDistance = FMath::Clamp(TargetDistance, 0.0f, SplineLength);

    const FVector TargetLocation =
        Spline->GetLocationAtDistanceAlongSpline(
            TargetDistance,
            ESplineCoordinateSpace::World
        );

    // Direction to target in local space
    const FVector LocalTarget =
        VehicleTransform.InverseTransformPosition(TargetLocation);

    const float HeadingError =
        FMath::Atan2(LocalTarget.Y, LocalTarget.X);

    float Steering =
        FMath::Clamp(SteeringGain * HeadingError, -1.0f, 1.0f);

    // Curvature-based speed reduction
    const float SampleAhead1 = FMath::Clamp(ClosestDistance + 50.0f, 0.0f, SplineLength);
    const float SampleAhead2 = FMath::Clamp(ClosestDistance + 250.0f, 0.0f, SplineLength);

    const FVector Tangent1 =
        Spline->GetDirectionAtDistanceAlongSpline(
            SampleAhead1,
            ESplineCoordinateSpace::World
        );

    const FVector Tangent2 =
        Spline->GetDirectionAtDistanceAlongSpline(
            SampleAhead2,
            ESplineCoordinateSpace::World
        );

    const float Dot =
        FMath::Clamp(FVector::DotProduct(Tangent1, Tangent2), -1.0f, 1.0f);

    const float TurnAngle = FMath::Acos(Dot);

    // Use a larger "max turn" threshold and a gentler falloff
    const float MaxTurnAngle = PI / 3.0f; // 60° instead of 45°
    const float TurnSharpness = FMath::Clamp(TurnAngle / MaxTurnAngle, 0.0f, 1.0f);

    // Instead of multiplying speed by (1 - sharpness), use a milder curve
    const float SpeedFactor = 1.0f - 0.5f * TurnSharpness; // at worst, 50% reduction

    const float TargetSpeedKmh =
        FMath::Clamp(
            SpeedKmh * SpeedFactor,
            MinSpeedKmh,
            MaxSpeedKmh
        );

    const float TargetSpeedCmPerSecond = (TargetSpeedKmh * 100.0f) / 3.6f;

    const float CurrentForwardSpeed = GetForwardSpeedCmPerSecond();
    const float SpeedError = TargetSpeedCmPerSecond - CurrentForwardSpeed;

    // Launch phase (full throttle up to LaunchUntilFraction of the target), then PID to hold the speed.
    float Throttle = 0.0f;
    const bool bNearRouteEnd = (SplineLength - ClosestDistance) < 300.0f;   // leave the end-of-route braking alone
    if (!bNearRouteEnd && CurrentForwardSpeed < LaunchUntilFraction * TargetSpeedCmPerSecond)
    {
        Throttle = LaunchThrottle;
        SpeedPID.Prime(SpeedError);   // no integral windup, no derivative kick at the hand-over
    }
    else
    {
        Throttle = SpeedPID.Update(SpeedError, DeltaTime, 0.0f, 1.0f);
    }

    // Optional: simple brake logic (only if significantly above target)
    float Brake = 0.0f;
    if (CurrentForwardSpeed > TargetSpeedCmPerSecond + 30.0f)
    {
        Brake = FMath::Clamp((CurrentForwardSpeed - TargetSpeedCmPerSecond) * 0.001f, 0.0f, 1.0f);
    }

    // Near the end of the spline, reduce throttle and gently brake
    const float DistanceToEnd = SplineLength - ClosestDistance;
    float BrakeZoneCm = 300.0f;

    if (SpeedKmh <= 5.0f)
    {
        BrakeZoneCm = 50.0f;
    }
    else if (SpeedKmh <= 10.0f)
    {
        BrakeZoneCm = 150.0f;
    }
    else if (SpeedKmh <= 15.0f)
    {
        BrakeZoneCm = 300.0f;
    }
    else
    {
        const float SpeedRatio = SpeedKmh / 15.0f;

        BrakeZoneCm = 300.0f * SpeedRatio * SpeedRatio;
    }

    if (DistanceToEnd < BrakeZoneCm)
    {
        const float EndBrakeFactor = 1.0f - (DistanceToEnd / BrakeZoneCm);
        Brake = FMath::Max(Brake, EndBrakeFactor * 0.5f);
        Throttle = Throttle * (1.0f - EndBrakeFactor);
    }

    const float RouteCompletionToleranceCm = 5.0f;

    // Very close to the end: cut throttle, minimal brake, no steering
    if (DistanceToEnd < RouteCompletionToleranceCm)
    {
        bReachedEnd = true;
        bFollowing = false;

        Throttle = 0.0f;
        Brake = 0.0f; // let handbrake hold it when we stop following
        Steering = 0.0f;

        SpeedPID.Reset();
    }

    VehicleMove->SetThrottleInput(Throttle);
    VehicleMove->SetBrakeInput(Brake);
    VehicleMove->SetSteeringInput(Steering * -1); // -1 for rear wheel steering

    // Handbrake only used in StopFollowing/ResetFollower or when reaching the end, not during normal driving
    VehicleMove->SetHandbrakeInput(
        bReachedEnd ? 1.0f : 0.0f
    );
}

float UForkliftRouteFollower::GetForwardSpeedCmPerSecond() const
{
    AActor* Owner = GetOwner();
    if (!Owner)
    {
        return 0.0f;
    }

    const FVector Vel = Owner->GetVelocity();
    const FVector Forward = Owner->GetActorForwardVector();
    return FVector::DotProduct(Vel, Forward);
}

UChaosWheeledVehicleMovementComponent* UForkliftRouteFollower::GetVehicleMovement() const
{
    AActor* Owner = GetOwner();
    if (!Owner)
    {
        return nullptr;
    }

    return Cast<UChaosWheeledVehicleMovementComponent>(
        Owner->FindComponentByClass<UChaosWheeledVehicleMovementComponent>()
    );
}

void UForkliftRouteFollower::EmergencyStop(float BrakeInput)
{
    UE_LOG(LogTemp, Warning, TEXT("ForkliftRouteFollower: EmergencyStop (brake input %.3f)"), BrakeInput);

    bFollowing = false;
    bEmergencyStop = true;
    EmergencyBrakeInput = FMath::Clamp(BrakeInput, 0.0f, 1.0f);
    SpeedPID.Reset();

    if (UChaosWheeledVehicleMovementComponent* VehicleMove = GetVehicleMovement())
    {
        VehicleMove->SetThrottleInput(0.0f);
        VehicleMove->SetSteeringInput(0.0f);
        VehicleMove->SetHandbrakeInput(false);
        VehicleMove->SetBrakeInput(EmergencyBrakeInput);
    }
}

bool UForkliftRouteFollower::HasReachedEnd() const
{
    return bReachedEnd;
}
