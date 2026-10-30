#pragma once

#include "CoreMinimal.h"
#include "ScenarioTypes.generated.h"

UENUM(BlueprintType)
enum class EScenarioType : uint8
{
    BaselineVisible       UMETA(DisplayName = "Baseline Visible"),
    LateralCrossing       UMETA(DisplayName = "Lateral Crossing"),
    OccludedEmergence     UMETA(DisplayName = "Occluded Emergence"),
    PartialVisibility     UMETA(DisplayName = "Partial Visibility"),
    MultiplePedestrians   UMETA(DisplayName = "Multiple Pedestrians"),
    StationaryNearPath    UMETA(DisplayName = "Stationary Near Path"),
    RelativeMotion        UMETA(DisplayName = "Relative Motion"),
    NearMiss              UMETA(DisplayName = "Near Miss")
};

/** When a pedestrian starts walking. */
UENUM(BlueprintType)
enum class EPedestrianStartMode : uint8
{
    /** StartDelaySeconds after capture starts (original behavior). */
    Delay            UMETA(DisplayName = "Time Delay"),
    /** When the forklift front is StartDistanceM from this pedestrian's conflict point. */
    ForkliftDistance UMETA(DisplayName = "Forklift Distance"),
    /**
     * Timed so the pedestrian reaches the conflict point ArrivalOffsetS seconds BEFORE the forklift would
     * (at its target speed). 0 = collision course; > 0 = pedestrian arrives earlier (more time to detect);
     * < 0 = pedestrian arrives after the forklift has passed (non-conflict / near miss).
     * The same value means the same situation at every forklift speed.
     */
    ArrivalOffset    UMETA(DisplayName = "Arrival Offset")
};

USTRUCT(BlueprintType)
struct FScenarioPedestrianConfig
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
    FName StartPointId;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
    FVector StartLocation = FVector::ZeroVector;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
    FRotator StartRotation = FRotator::ZeroRotator;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
    TArray<FName> WaypointIds;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
    TArray<FVector> Waypoints;

    /** cm/s. 139 = Euro NCAP adult walking speed (5 km/h); running worker (S4) uses 220 (Euro NCAP far-side adult, 8 km/h). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
    float MovementSpeed = 139.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
    float StartDelaySeconds = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
    bool bIsStationary = false;

    /**
     * Pedestrian is crouched for the whole run (e.g. a worker crouching at a pallet). Sets bScenarioCrouched on the
     * APedestrianCharacter at reset; the Animation Blueprint reads it to pick the crouch pose. The capsule is not
     * changed, so collision, Stop on Overlap and the clearance evaluation behave exactly as for a standing person.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian")
    bool bCrouched = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian|Start")
    EPedestrianStartMode StartMode = EPedestrianStartMode::ArrivalOffset;

    /**
     * Where the pedestrian's path meets the forklift's lane.
     * -1 (default) = automatic: the point of the pedestrian's path closest to the forklift route
     *                (for a pedestrian without waypoints: its own location).
     * >= 0         = index into Waypoints/WaypointIds, to force a specific waypoint.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian|Start")
    int32 ConflictWaypointIndex = -1;

    /** ForkliftDistance mode: forklift front distance to the conflict point (m) at which the pedestrian starts. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian|Start")
    float StartDistanceM = 10.0f;

    /** ArrivalOffset mode: seconds the pedestrian reaches the conflict point before the forklift would. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pedestrian|Start")
    float ArrivalOffsetS = 0.0f;
};