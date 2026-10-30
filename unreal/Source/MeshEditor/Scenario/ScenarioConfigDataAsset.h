#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "ScenarioTypes.h"
#include "ScenarioConfigDataAsset.generated.h"

UCLASS(BlueprintType)
class MESHEDITOR_API UScenarioConfigDataAsset : public UDataAsset
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Scenario")
    FName ScenarioId;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Scenario")
    EScenarioType ScenarioType = EScenarioType::BaselineVisible;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Scenario")
    float WarmupTimeSeconds = 2.0f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Scenario")
    float DurationSeconds = 10.0f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Forklift")
    FName ForkliftRouteId;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Forklift")
    FName ForkliftStartPointId;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Forklift")
    FVector ForkliftStartLocation = FVector::ZeroVector;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Forklift")
    FRotator ForkliftStartRotation = FRotator::ZeroRotator;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Forklift")
    float ForkliftSpeedKmh = 3.0f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pedestrians")
    TArray<FScenarioPedestrianConfig> Pedestrians;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Occlusion")
    bool bUseOccluder = false;

    /**
     * With bUseOccluder: show only the ScenarioManager occluders that carry this Actor Tag (Details > Actor > Tags),
     * e.g. "Load_1_2m" or "Load_1_8m". None = show every assigned occluder.
     */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Occlusion")
    FName OccluderActorTag;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Output")
    bool bSaveSampleImages = true;

    // ---------------- Variants (expanded automatically by the ScenarioManager batch) ----------------

    /**
     * Forklift speeds to run this scenario at, e.g. [5, 10, 15]. Empty = only ForkliftSpeedKmh.
     * Each speed becomes its own variant (own run folders, e.g. "LateralCrossing_v10").
     */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Variants")
    TArray<float> VariantForkliftSpeedsKmh;

    /**
     * Timing shifts (s) added to every ArrivalOffset pedestrian's ArrivalOffsetS, e.g. [2, 1, 0.5] for
     * early / base / late. Empty = no shift. Combined with every speed variant.
     */
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Variants")
    TArray<float> VariantTimingShiftsS;
};