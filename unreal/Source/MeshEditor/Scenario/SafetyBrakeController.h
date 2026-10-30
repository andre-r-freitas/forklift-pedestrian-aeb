#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SafetyBrakeController.generated.h"

class UForkliftRouteFollower;

/** One detected person, already converted to the forklift's frame. */
struct FPersonObservation
{
	bool bDepthValid = false;
	float ForwardM = 0.0f;      // ahead of the forklift's FRONT edge (negative = beside/behind the front)
	float LateralM = 0.0f;      // + right, - left of the forklift's center line
	float Confidence = 0.0f;
};

UENUM(BlueprintType)
enum class ESafetyBrakeState : uint8
{
	Idle,        // no confirmed hazard
	Pending,     // hazard confirmed, waiting for the latency L to pass
	Braking,     // emergency brake applied
	Stopped      // forklift stopped and held
};

/**
 * Zone-based automatic emergency braking for the forklift (closed loop).
 *
 * Every perception frame, a person counts as a hazard if they are inside the forklift's path corridor
 * (|lateral| < CorridorHalfWidthM) and closer than the trigger distance:
 *
 *     trigger = v * (ReactionLatencyS + DetectionDelay + BrakeDeadTimeS) + v^2 / (2 * AssumedDecelerationMps2) + SafetyMarginM
 *
 * DetectionDelay = ConfirmFrames x perception period (worst case: the person enters the zone just after a frame,
 * is first seen one period later and confirmed ConfirmFrames-1 periods after that). The period is measured from
 * the capture times (PerceptionPeriodS until two frames have been seen). Set bIncludeDetectionDelay = false to get
 * the original formula (latency + stopping distance + margin only).
 *
 * The hazard must be seen in ConfirmFrames of the last WindowFrames perception frames. The brake command is
 * then applied ReactionLatencyS after the CAPTURE time of the confirming frame (explicit, controlled latency),
 * using a partial brake input calibrated with the dynamics test to give a realistic forklift deceleration.
 * Once braking, the forklift stops and is held until the scenario is reset.
 *
 * Add this component to the forklift. It does nothing while bEnabled is false.
 */
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class MESHEDITOR_API USafetyBrakeController : public UActorComponent
{
	GENERATED_BODY()

public:
	USafetyBrakeController();

	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	/** Called by the inference actor once per processed frame. */
	void OnPerception(double CaptureTimeSeconds, const TArray<FPersonObservation>& Observations);

	/** Clears history and state; called by the ScenarioManager on every reset. */
	UFUNCTION(BlueprintCallable, Category = "Safety")
	void ResetController();

	/** Returns true if the observation is inside the current braking zone. */
	bool IsInZone(const FPersonObservation& Observation) const;

	UFUNCTION(BlueprintPure, Category = "Safety")
	ESafetyBrakeState GetState() const { return State; }

	UFUNCTION(BlueprintPure, Category = "Safety")
	float GetTriggerDistanceM() const;

	/** Game time at which the forklift came to a stop after an emergency brake (-1 = not stopped). */
	double GetStoppedTime() const { return StoppedTime; }

	/** Perception period used in the trigger distance (measured, or PerceptionPeriodS before two frames). */
	float GetPerceptionPeriodS() const;

	/** JSON key/value pairs (no braces) describing settings and what happened in this run. */
	FString BuildMetadataFields() const;

	// ---------------- Settings ----------------

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety")
	bool bEnabled = false;

	/**
	 * Half the forklift body width (1.73 m / 2) + pedestrian radius (0.42 m) + ~0.2 m for position-estimate error.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Zone")
	float CorridorHalfWidthM = 1.5f;

	/** Distance from the forklift actor's origin to its front edge (forks/mast), along its forward axis. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Zone")
	float FrontOffsetFromOriginM = 2.58f;   // measured: half body length 1.23 m + fork 1.35 m

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Zone")
	float SafetyMarginM = 1.0f;

	/**
	 * Deceleration assumed when computing the trigger distance. Target ~0.6 g (sourced forklift value);
	 * dynamics test at EmergencyBrakeInput 1.0: steady deceleration 5.3-6.6 m/s^2 (5/10/15 km/h), mean ~5.99.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Braking")
	float AssumedDecelerationMps2 = 5.97f;

	/** Brake input (0-1) used for the emergency stop (calibrated with the dynamics test). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Braking", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float EmergencyBrakeInput = 1.0f;

	/** Explicit latency L between frame capture and the brake command. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Braking", meta = (ClampMin = "0.0"))
	float ReactionLatencyS = 0.1f;

	/** Time between the brake command and full deceleration (dynamics test: dead_time_s, mean 0.104 s). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Braking", meta = (ClampMin = "0.0"))
	float BrakeDeadTimeS = 0.1f;

	/** Adds the detection/confirmation delay and brake dead time to the trigger distance (see class comment). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Braking")
	bool bIncludeDetectionDelay = true;

	/** Perception period assumed until it can be measured from two consecutive frames (= inference interval). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Braking", meta = (ClampMin = "0.01"))
	float PerceptionPeriodS = 0.2f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Confirmation", meta = (ClampMin = "1"))
	int32 ConfirmFrames = 2;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Confirmation", meta = (ClampMin = "1"))
	int32 WindowFrames = 3;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Safety|Confirmation")
	float MinConfidence = 0.25f;

	// ---------------- Last-frame info (for logging) ----------------

	bool bLastFrameHazard = false;
	float LastClosestHazardForwardM = -1.0f;

protected:
	virtual void BeginPlay() override;

private:
	float GetForwardSpeedMps() const;
	void EngageBrake();

	UForkliftRouteFollower* GetRouteFollower() const;

	ESafetyBrakeState State = ESafetyBrakeState::Idle;
	TArray<bool> HazardHistory;

	double ScheduledBrakeTime = -1.0;
	double ConfirmCaptureTime = -1.0;
	double BrakeAppliedTime = -1.0;
	double StoppedTime = -1.0;
	float SpeedAtConfirmKmh = -1.0f;
	float SpeedAtBrakeKmh = -1.0f;
	float TriggerDistanceAtConfirmM = -1.0f;
	float HazardDistanceAtConfirmM = -1.0f;
	FVector BrakeStartLocation = FVector::ZeroVector;
	float BrakingDistanceM = -1.0f;
	int32 HazardFramesTotal = 0;
	double LastPerceptionCaptureTime = -1.0;
	float MeasuredPerceptionPeriodS = -1.0f;
};
