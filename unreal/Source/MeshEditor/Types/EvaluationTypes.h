#pragma once

#include "CoreMinimal.h"
#include "EvaluationTypes.generated.h"

USTRUCT(BlueprintType)
struct FFrameRunStats
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly)
	int32 FrameIndex = -1;

	UPROPERTY(BlueprintReadOnly)
	FString RunId;

	UPROPERTY(BlueprintReadOnly)
	FString ScenarioName;

	UPROPERTY(BlueprintReadOnly)
	float TimestampSeconds = 0.0f;

	UPROPERTY(BlueprintReadOnly)
	float InferenceTimeMs = 0.0f;

	UPROPERTY(BlueprintReadOnly)
	int32 NumDetections = 0;

	UPROPERTY(BlueprintReadOnly)
	int32 NumGroundTruth = 0;

	UPROPERTY(BlueprintReadOnly)
	int32 NumTruePositives = 0;

	UPROPERTY(BlueprintReadOnly)
	int32 NumFalsePositives = 0;

	UPROPERTY(BlueprintReadOnly)
	int32 NumFalseNegatives = 0;

	UPROPERTY(BlueprintReadOnly)
	float MeanMatchedIoU = 0.0f;

	UPROPERTY(BlueprintReadOnly)
	bool bSavedSampleImage = false;
};
