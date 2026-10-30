#pragma once

#include "CoreMinimal.h"
#include "Types/DetectionTypes.h"
#include "GroundTruthTypes.generated.h"

USTRUCT(BlueprintType)
struct FGroundTruthInstanceInfo
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Ground Truth")
	int32 StencilID = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Ground Truth")
	FName ActorName = NAME_None;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Ground Truth")
	int32 ClassID = -1;
};

USTRUCT(BlueprintType)
struct FGroundTruthAssignmentResult
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Ground Truth")
	TArray<FGroundTruthInstanceInfo> Instances;

	TMap<int32, int32> StencilIDToClassID;
	TMap<int32, FName> StencilIDToActorName;

	// NEW: actor per stencil ID, used to log world state (position, velocity) per pedestrian.
	TMap<int32, TWeakObjectPtr<AActor>> StencilIDToActor;

	bool IsValid() const
	{
		return Instances.Num() > 0;
	}
};

USTRUCT(BlueprintType)
struct FGroundTruthExtractionResult
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Ground Truth")
	TArray<FBoundingBox2> Boxes;

	// Sorted list of all stencil IDs seen in the frame (kept for compatibility).
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Ground Truth")
	TArray<int32> ObservedStencilIDs;

	// NEW: the following arrays are parallel to Boxes (same index = same pedestrian).

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Ground Truth")
	TArray<int32> BoxStencilIDs;

	// Pixels of the pedestrian that are actually visible (R channel, depth-tested stencil).
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Ground Truth")
	TArray<int32> VisiblePixelCounts;

	// Pixels of the full silhouette including occluded parts (G channel), or -1 when not available.
	// Visibility fraction = VisiblePixelCounts / FullPixelCounts.
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Ground Truth")
	TArray<int32> FullPixelCounts;
};
