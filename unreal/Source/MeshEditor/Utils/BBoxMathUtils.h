#pragma once

#include "CoreMinimal.h"
#include "Types/DetectionTypes.h"
#include "Types/EvaluationTypes.h"

namespace BBoxMathUtils
{
	float ComputeIoU(const FBoundingBox2& BoxA, const FBoundingBox2& BoxB);

	TArray<FBoundingBox2> ApplyNMS(const TArray<FBoundingBox2>& Detections, float IoUThreshold = 0.45f);

	TArray<FBoundingBox2> FilterBoxesByClass(const TArray<FBoundingBox2>& Boxes, int32 AllowedClassID);

	FFrameRunStats EvaluateFrame(
		int32 FrameIndex,
		const FString& RunId,
		const FString& ScenarioName,
		float TimestampSeconds,
		float InferenceTimeMs,
		float MatchIoUThreshold,
		const TArray<FBoundingBox2>& Detections,
		const TArray<FBoundingBox2>& GroundTruth,
		int32 EvaluatedClassID = INDEX_NONE,
		TArray<int32>* OutDetectionToGT = nullptr,   // NEW: per detection, index of the matched GT box or INDEX_NONE
		TArray<float>* OutDetectionIoU = nullptr);   // NEW: per detection, IoU with the matched GT box (0 if none)
}

