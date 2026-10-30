#include "SafetyBrakeController.h"

#include "ForkliftRouteFollower.h"
#include "GameFramework/Actor.h"
#include "Engine/World.h"

USafetyBrakeController::USafetyBrakeController()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void USafetyBrakeController::BeginPlay()
{
	Super::BeginPlay();
	ResetController();
}

void USafetyBrakeController::ResetController()
{
	State = ESafetyBrakeState::Idle;
	HazardHistory.Reset();
	ScheduledBrakeTime = -1.0;
	ConfirmCaptureTime = -1.0;
	BrakeAppliedTime = -1.0;
	StoppedTime = -1.0;
	SpeedAtConfirmKmh = -1.0f;
	SpeedAtBrakeKmh = -1.0f;
	TriggerDistanceAtConfirmM = -1.0f;
	HazardDistanceAtConfirmM = -1.0f;
	BrakeStartLocation = FVector::ZeroVector;
	BrakingDistanceM = -1.0f;
	HazardFramesTotal = 0;
	LastPerceptionCaptureTime = -1.0;
	MeasuredPerceptionPeriodS = -1.0f;
	bLastFrameHazard = false;
	LastClosestHazardForwardM = -1.0f;
}

float USafetyBrakeController::GetForwardSpeedMps() const
{
	const AActor* Owner = GetOwner();
	if (!Owner)
	{
		return 0.0f;
	}
	return static_cast<float>(FVector::DotProduct(Owner->GetVelocity(), Owner->GetActorForwardVector()) / 100.0);
}

float USafetyBrakeController::GetPerceptionPeriodS() const
{
	return MeasuredPerceptionPeriodS > 0.0f ? MeasuredPerceptionPeriodS : PerceptionPeriodS;
}

float USafetyBrakeController::GetTriggerDistanceM() const
{
	const float V = FMath::Max(0.0f, GetForwardSpeedMps());
	const float Decel = FMath::Max(0.1f, AssumedDecelerationMps2);

	float DelayS = ReactionLatencyS;
	if (bIncludeDetectionDelay)
	{
		DelayS += ConfirmFrames * GetPerceptionPeriodS() + BrakeDeadTimeS;
	}

	return V * DelayS + (V * V) / (2.0f * Decel) + SafetyMarginM;
}

bool USafetyBrakeController::IsInZone(const FPersonObservation& Observation) const
{
	return Observation.bDepthValid
		&& Observation.Confidence >= MinConfidence
		&& FMath::Abs(Observation.LateralM) <= CorridorHalfWidthM
		&& Observation.ForwardM > -0.5f                        // ignore people clearly behind the front edge
		&& Observation.ForwardM <= GetTriggerDistanceM();
}

void USafetyBrakeController::OnPerception(double CaptureTimeSeconds, const TArray<FPersonObservation>& Observations)
{
	bLastFrameHazard = false;
	LastClosestHazardForwardM = -1.0f;

	if (LastPerceptionCaptureTime >= 0.0 && CaptureTimeSeconds > LastPerceptionCaptureTime)
	{
		MeasuredPerceptionPeriodS = static_cast<float>(CaptureTimeSeconds - LastPerceptionCaptureTime);
	}
	LastPerceptionCaptureTime = CaptureTimeSeconds;

	for (const FPersonObservation& Obs : Observations)
	{
		if (IsInZone(Obs))
		{
			bLastFrameHazard = true;
			if (LastClosestHazardForwardM < 0.0f || Obs.ForwardM < LastClosestHazardForwardM)
			{
				LastClosestHazardForwardM = Obs.ForwardM;
			}
		}
	}

	if (bLastFrameHazard)
	{
		++HazardFramesTotal;
	}

	HazardHistory.Add(bLastFrameHazard);
	while (HazardHistory.Num() > WindowFrames)
	{
		HazardHistory.RemoveAt(0);
	}

	if (!bEnabled || State != ESafetyBrakeState::Idle)
	{
		return;
	}

	int32 Hits = 0;
	for (bool bHit : HazardHistory)
	{
		Hits += bHit ? 1 : 0;
	}

	if (Hits >= ConfirmFrames && bLastFrameHazard)
	{
		State = ESafetyBrakeState::Pending;
		ConfirmCaptureTime = CaptureTimeSeconds;
		ScheduledBrakeTime = CaptureTimeSeconds + ReactionLatencyS;
		SpeedAtConfirmKmh = GetForwardSpeedMps() * 3.6f;
		TriggerDistanceAtConfirmM = GetTriggerDistanceM();
		HazardDistanceAtConfirmM = LastClosestHazardForwardM;

		UE_LOG(LogTemp, Warning, TEXT("Safety: hazard confirmed at %.2f m (trigger %.2f m, %.1f km/h). Brake at t=%.3f"),
			HazardDistanceAtConfirmM, TriggerDistanceAtConfirmM, SpeedAtConfirmKmh, ScheduledBrakeTime);
	}
}

void USafetyBrakeController::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	const UWorld* World = GetWorld();
	if (!World || !GetOwner())
	{
		return;
	}

	const double Now = World->GetTimeSeconds();

	if (State == ESafetyBrakeState::Pending && Now >= ScheduledBrakeTime)
	{
		EngageBrake();
	}
	else if (State == ESafetyBrakeState::Braking)
	{
		if (GetForwardSpeedMps() <= 0.01f)
		{
			State = ESafetyBrakeState::Stopped;
			StoppedTime = Now;
			BrakingDistanceM = static_cast<float>(FVector::Dist2D(BrakeStartLocation, GetOwner()->GetActorLocation()) / 100.0);
			UE_LOG(LogTemp, Warning, TEXT("Safety: forklift stopped after %.2f m, %.2f s."), BrakingDistanceM, StoppedTime - BrakeAppliedTime);
		}
	}
}

void USafetyBrakeController::EngageBrake()
{
	const UWorld* World = GetWorld();
	BrakeAppliedTime = World ? World->GetTimeSeconds() : 0.0;
	SpeedAtBrakeKmh = GetForwardSpeedMps() * 3.6f;
	BrakeStartLocation = GetOwner()->GetActorLocation();
	State = ESafetyBrakeState::Braking;

	if (UForkliftRouteFollower* Follower = GetRouteFollower())
	{
		Follower->EmergencyStop(EmergencyBrakeInput);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Safety: no UForkliftRouteFollower on the forklift; cannot brake."));
	}
}

UForkliftRouteFollower* USafetyBrakeController::GetRouteFollower() const
{
	return GetOwner() ? GetOwner()->FindComponentByClass<UForkliftRouteFollower>() : nullptr;
}

FString USafetyBrakeController::BuildMetadataFields() const
{
	const UEnum* StateEnum = StaticEnum<ESafetyBrakeState>();

	return FString::Printf(
		TEXT("\"safety\": {\n")
		TEXT("    \"enabled\": %s,\n")
		TEXT("    \"corridor_half_width_m\": %.2f,\n")
		TEXT("    \"front_offset_from_origin_m\": %.2f,\n")
		TEXT("    \"safety_margin_m\": %.2f,\n")
		TEXT("    \"assumed_deceleration_mps2\": %.2f,\n")
		TEXT("    \"emergency_brake_input\": %.3f,\n")
		TEXT("    \"reaction_latency_s\": %.3f,\n")
		TEXT("    \"brake_dead_time_s\": %.3f,\n")
		TEXT("    \"include_detection_delay\": %s,\n")
		TEXT("    \"perception_period_s\": %.3f,\n")
		TEXT("    \"confirm_frames\": %d,\n")
		TEXT("    \"window_frames\": %d,\n")
		TEXT("    \"min_confidence\": %.2f,\n")
		TEXT("    \"final_state\": \"%s\",\n")
		TEXT("    \"hazard_frames_total\": %d,\n")
		TEXT("    \"confirm_capture_time_s\": %.4f,\n")
		TEXT("    \"hazard_distance_at_confirm_m\": %.3f,\n")
		TEXT("    \"trigger_distance_at_confirm_m\": %.3f,\n")
		TEXT("    \"speed_at_confirm_kmh\": %.2f,\n")
		TEXT("    \"brake_applied_time_s\": %.4f,\n")
		TEXT("    \"speed_at_brake_kmh\": %.2f,\n")
		TEXT("    \"stopped_time_s\": %.4f,\n")
		TEXT("    \"braking_distance_m\": %.3f\n")
		TEXT("  }"),
		bEnabled ? TEXT("true") : TEXT("false"),
		CorridorHalfWidthM,
		FrontOffsetFromOriginM,
		SafetyMarginM,
		AssumedDecelerationMps2,
		EmergencyBrakeInput,
		ReactionLatencyS,
		BrakeDeadTimeS,
		bIncludeDetectionDelay ? TEXT("true") : TEXT("false"),
		GetPerceptionPeriodS(),
		ConfirmFrames,
		WindowFrames,
		MinConfidence,
		StateEnum ? *StateEnum->GetNameStringByValue(static_cast<int64>(State)) : TEXT(""),
		HazardFramesTotal,
		ConfirmCaptureTime,
		HazardDistanceAtConfirmM,
		TriggerDistanceAtConfirmM,
		SpeedAtConfirmKmh,
		BrakeAppliedTime,
		SpeedAtBrakeKmh,
		StoppedTime,
		BrakingDistanceM);
}
