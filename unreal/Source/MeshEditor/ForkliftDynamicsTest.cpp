#include "ForkliftDynamicsTest.h"

#include "ChaosWheeledVehicleMovementComponent.h"
#include "Components/PrimitiveComponent.h"
#include "GameFramework/Actor.h"
#include "HAL/FileManager.h"
#include "Misc/DateTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

UForkliftDynamicsTest::UForkliftDynamicsTest()
{
    PrimaryComponentTick.bCanEverTick = true;
}

void UForkliftDynamicsTest::BeginPlay()
{
    Super::BeginPlay();

    if (bAutoStartOnBeginPlay)
    {
        StartTest();
    }
}

void UForkliftDynamicsTest::StartTest()
{
    AActor* Owner = GetOwner();
    if (!Owner || !GetVehicle())
    {
        UE_LOG(LogTemp, Error, TEXT("DynamicsTest: owner has no UChaosWheeledVehicleMovementComponent."));
        return;
    }

    if (TargetSpeedsKmh.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("DynamicsTest: no target speeds configured."));
        return;
    }

    StartTransform = Owner->GetActorTransform();

    OutputFolder = FPaths::ProjectSavedDir() / TEXT("VehicleTests") /
        FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
    IFileManager::Get().MakeDirectory(*OutputFolder, true);

    TickBuffer = TEXT("run,target_kmh,phase,t_run_s,x_m,y_m,speed_kmh,throttle,brake,handbrake,gear,engine_rpm\n");
    SummaryBuffer = TEXT("run,target_kmh,stop_input,time_to_target_s,dist_to_target_m,speed_at_brake_kmh,stop_time_s,stop_dist_m,mean_decel_mps2,equiv_decel_mps2,dead_time_s,steady_decel_mps2,timed_out\n");
    FlushLogs(); // writes the headers

    RunIndex = 0;
    TotalRuns = TargetSpeedsKmh.Num() * RepetitionsPerSpeed;

    UE_LOG(LogTemp, Warning, TEXT("DynamicsTest: starting %d runs, output in %s"), TotalRuns, *OutputFolder);

    BeginRun();
}

void UForkliftDynamicsTest::BeginRun()
{
    AActor* Owner = GetOwner();

    // Physics-aware teleport back to the start, velocities cleared.
    Owner->SetActorTransform(StartTransform, false, nullptr, ETeleportType::TeleportPhysics);

    if (UPrimitiveComponent* Root = Cast<UPrimitiveComponent>(Owner->GetRootComponent()))
    {
        if (Root->IsSimulatingPhysics())
        {
            Root->SetPhysicsLinearVelocity(FVector::ZeroVector);
            Root->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
        }
    }

    TimeToTargetS = -1.0f;
    DistToTargetM = -1.0f;
    SpeedAtBrakeKmh = 0.0f;

    RunStartTime = Now();
    EnterPhase(EDynamicsTestPhase::Settling);
}

void UForkliftDynamicsTest::EnterPhase(EDynamicsTestPhase NewPhase)
{
    Phase = NewPhase;
    PhaseStartTime = Now();

    const FVector Location = GetOwner()->GetActorLocation();

    if (NewPhase == EDynamicsTestPhase::Accelerating)
    {
        AccelStartLocation = Location;
        AccelStartTime = Now();
    }
    else if (NewPhase == EDynamicsTestPhase::Braking)
    {
        BrakeStartLocation = Location;
        BrakeStartTime = Now();
        SpeedAtBrakeKmh = GetSpeedKmh();
        BrakeDeadTimeS = -1.0f;
        TimeAt80PctS = -1.0f;
        TimeAt20PctS = -1.0f;
    }
}

void UForkliftDynamicsTest::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (Phase == EDynamicsTestPhase::Idle || Phase == EDynamicsTestPhase::Done || !GetVehicle())
    {
        return;
    }

    const float Speed = GetSpeedKmh();
    const float Target = GetCurrentTargetKmh();
    const float PhaseElapsed = Now() - PhaseStartTime;

    if (Phase != EDynamicsTestPhase::Settling && PhaseElapsed > PhaseTimeoutSeconds)
    {
        UE_LOG(LogTemp, Warning, TEXT("DynamicsTest: run %d timed out in phase %s"), RunIndex, PhaseName(Phase));
        FinishRun(true);
        return;
    }

    switch (Phase)
    {
    case EDynamicsTestPhase::Settling:
        ApplyInputs(0.0f, 0.0f, 1.0f);
        if (PhaseElapsed >= SettleSeconds)
        {
            EnterPhase(EDynamicsTestPhase::Accelerating);
        }
        break;

    case EDynamicsTestPhase::Accelerating:
        ApplyInputs(1.0f, 0.0f, 0.0f);
        if (Speed >= Target)
        {
            TimeToTargetS = Now() - AccelStartTime;
            DistToTargetM = DistanceMeters2D(AccelStartLocation, GetOwner()->GetActorLocation());
            EnterPhase(EDynamicsTestPhase::Cruising);
        }
        break;

    case EDynamicsTestPhase::Cruising:
    {
        const float Throttle = FMath::Clamp(CruiseThrottleGain * (Target - Speed), 0.0f, 1.0f);
        ApplyInputs(Throttle, 0.0f, 0.0f);
        if (PhaseElapsed >= CruiseSeconds)
        {
            EnterPhase(EDynamicsTestPhase::Braking);
        }
        break;
    }

    case EDynamicsTestPhase::Braking:
        if (bUseHandbrakeForStop)
        {
            ApplyInputs(0.0f, 0.0f, 1.0f);
        }
        else
        {
            ApplyInputs(0.0f, StopBrakeInput, 0.0f);
        }

        {
            const float Elapsed = Now() - BrakeStartTime;
            if (BrakeDeadTimeS < 0.0f && Speed < 0.98f * SpeedAtBrakeKmh) { BrakeDeadTimeS = Elapsed; }
            if (TimeAt80PctS < 0.0f && Speed <= 0.8f * SpeedAtBrakeKmh) { TimeAt80PctS = Elapsed; }
            if (TimeAt20PctS < 0.0f && Speed <= 0.2f * SpeedAtBrakeKmh) { TimeAt20PctS = Elapsed; }
        }

        // Signed speed: also catches Chaos "reverse as brake" kicking in after the stop.
        if (Speed <= StoppedSpeedKmh)
        {
            AppendTickLine();
            FinishRun(false);
            return;
        }
        break;

    default:
        break;
    }

    AppendTickLine();
}

void UForkliftDynamicsTest::FinishRun(bool bTimedOut)
{
    const float StopTimeS = (Phase == EDynamicsTestPhase::Braking && !bTimedOut) ? Now() - BrakeStartTime : -1.0f;
    const float StopDistM = (StopTimeS > 0.0f) ? DistanceMeters2D(BrakeStartLocation, GetOwner()->GetActorLocation()) : -1.0f;

    const float V0 = SpeedAtBrakeKmh / 3.6f; // m/s
    const float MeanDecel = (StopTimeS > 0.0f) ? V0 / StopTimeS : -1.0f;
    const float EquivDecel = (StopDistM > 0.0f) ? (V0 * V0) / (2.0f * StopDistM) : -1.0f;
    // Steady deceleration: speed drop from 80% to 20% of V0, divided by the time it took.
    const float SteadyDecel = (TimeAt80PctS > 0.0f && TimeAt20PctS > TimeAt80PctS)
        ? (0.6f * V0) / (TimeAt20PctS - TimeAt80PctS) : -1.0f;

    SummaryBuffer += FString::Printf(
        TEXT("%d,%.1f,%s,%.3f,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n"),
        RunIndex,
        GetCurrentTargetKmh(),
        bUseHandbrakeForStop ? TEXT("handbrake") : *FString::Printf(TEXT("brake_%.2f"), StopBrakeInput),
        TimeToTargetS,
        DistToTargetM,
        SpeedAtBrakeKmh,
        StopTimeS,
        StopDistM,
        MeanDecel,
        EquivDecel,
        BrakeDeadTimeS,
        SteadyDecel,
        bTimedOut ? 1 : 0);

    UE_LOG(LogTemp, Warning,
        TEXT("DynamicsTest run %d | target %.1f km/h | 0-target %.2f s over %.2f m | brake from %.1f km/h: %.2f s, %.2f m, %.2f m/s^2 (steady %.2f, dead time %.3f s)%s"),
        RunIndex, GetCurrentTargetKmh(), TimeToTargetS, DistToTargetM, SpeedAtBrakeKmh, StopTimeS, StopDistM, EquivDecel, SteadyDecel, BrakeDeadTimeS,
        bTimedOut ? TEXT(" (TIMED OUT)") : TEXT(""));

    FlushLogs();

    ++RunIndex;
    if (RunIndex < TotalRuns)
    {
        BeginRun();
    }
    else
    {
        ApplyInputs(0.0f, 0.0f, 1.0f);
        Phase = EDynamicsTestPhase::Done;
        UE_LOG(LogTemp, Warning, TEXT("DynamicsTest: all runs done. Results in %s"), *OutputFolder);
    }
}

void UForkliftDynamicsTest::ApplyInputs(float InThrottle, float InBrake, float InHandbrake)
{
    UChaosWheeledVehicleMovementComponent* Vehicle = GetVehicle();
    if (!Vehicle)
    {
        return;
    }

    Vehicle->SetThrottleInput(InThrottle);
    Vehicle->SetBrakeInput(InBrake);
    Vehicle->SetHandbrakeInput(InHandbrake > 0.5f); // takes a bool in UE 5.6 (on/off, no partial handbrake)
    Vehicle->SetSteeringInput(0.0f);

    CurrentThrottle = InThrottle;
    CurrentBrake = InBrake;
    CurrentHandbrake = InHandbrake;
}

void UForkliftDynamicsTest::AppendTickLine()
{
    UChaosWheeledVehicleMovementComponent* Vehicle = GetVehicle();
    const FVector Location = GetOwner()->GetActorLocation();

    TickBuffer += FString::Printf(
        TEXT("%d,%.1f,%s,%.4f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%d,%.0f\n"),
        RunIndex,
        GetCurrentTargetKmh(),
        PhaseName(Phase),
        Now() - RunStartTime,
        Location.X / 100.0,
        Location.Y / 100.0,
        GetSpeedKmh(),
        CurrentThrottle,
        CurrentBrake,
        CurrentHandbrake,
        Vehicle ? Vehicle->GetCurrentGear() : 0,
        Vehicle ? Vehicle->GetEngineRotationSpeed() : 0.0f);
}

void UForkliftDynamicsTest::FlushLogs()
{
    if (!TickBuffer.IsEmpty())
    {
        FFileHelper::SaveStringToFile(TickBuffer, *(OutputFolder / TEXT("ticks.csv")),
            FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_Append);
        TickBuffer.Reset();
    }

    if (!SummaryBuffer.IsEmpty())
    {
        FFileHelper::SaveStringToFile(SummaryBuffer, *(OutputFolder / TEXT("summary.csv")),
            FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_Append);
        SummaryBuffer.Reset();
    }
}

UChaosWheeledVehicleMovementComponent* UForkliftDynamicsTest::GetVehicle() const
{
    AActor* Owner = GetOwner();
    return Owner ? Owner->FindComponentByClass<UChaosWheeledVehicleMovementComponent>() : nullptr;
}

float UForkliftDynamicsTest::GetSpeedKmh() const
{
    // Signed speed along the forklift's forward axis, cm/s -> km/h.
    const AActor* Owner = GetOwner();
    const float ForwardCmPerS = static_cast<float>(FVector::DotProduct(Owner->GetVelocity(), Owner->GetActorForwardVector()));
    return ForwardCmPerS * 0.036f;
}

float UForkliftDynamicsTest::GetCurrentTargetKmh() const
{
    const int32 SpeedIndex = FMath::Clamp(RunIndex / FMath::Max(1, RepetitionsPerSpeed), 0, TargetSpeedsKmh.Num() - 1);
    return TargetSpeedsKmh[SpeedIndex];
}

float UForkliftDynamicsTest::Now() const
{
    return GetWorld() ? GetWorld()->GetTimeSeconds() : 0.0f;
}

float UForkliftDynamicsTest::DistanceMeters2D(const FVector& A, const FVector& B)
{
    return static_cast<float>(FVector::Dist2D(A, B) / 100.0);
}

const TCHAR* UForkliftDynamicsTest::PhaseName(EDynamicsTestPhase InPhase)
{
    switch (InPhase)
    {
    case EDynamicsTestPhase::Idle:         return TEXT("idle");
    case EDynamicsTestPhase::Settling:     return TEXT("settling");
    case EDynamicsTestPhase::Accelerating: return TEXT("accelerating");
    case EDynamicsTestPhase::Cruising:     return TEXT("cruising");
    case EDynamicsTestPhase::Braking:      return TEXT("braking");
    case EDynamicsTestPhase::Done:         return TEXT("done");
    default:                               return TEXT("unknown");
    }
}