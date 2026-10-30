#include "BBoxMathUtils.h"

namespace BBoxMathUtils
{
	float ComputeIoU(const FBoundingBox2& BoxA, const FBoundingBox2& BoxB)
	{
		const float Ax1 = BoxA.CenterX - BoxA.Width * 0.5f;
		const float Ay1 = BoxA.CenterY - BoxA.Height * 0.5f;
		const float Ax2 = BoxA.CenterX + BoxA.Width * 0.5f;
		const float Ay2 = BoxA.CenterY + BoxA.Height * 0.5f;

		const float Bx1 = BoxB.CenterX - BoxB.Width * 0.5f;
		const float By1 = BoxB.CenterY - BoxB.Height * 0.5f;
		const float Bx2 = BoxB.CenterX + BoxB.Width * 0.5f;
		const float By2 = BoxB.CenterY + BoxB.Height * 0.5f;

		const float Ix1 = FMath::Max(Ax1, Bx1);
		const float Iy1 = FMath::Max(Ay1, By1);
		const float Ix2 = FMath::Min(Ax2, Bx2);
		const float Iy2 = FMath::Min(Ay2, By2);

		const float IW = FMath::Max(0.0f, Ix2 - Ix1);
		const float IH = FMath::Max(0.0f, Iy2 - Iy1);
		const float Intersection = IW * IH;

		const float AreaA = BoxA.Width * BoxA.Height;
		const float AreaB = BoxB.Width * BoxB.Height;
		const float Union = AreaA + AreaB - Intersection;

		return Union > 0.0f ? Intersection / Union : 0.0f;
	}

	TArray<FBoundingBox2> ApplyNMS(const TArray<FBoundingBox2>& Detections, float IoUThreshold)
	{
		if (Detections.Num() == 0)
		{
			return {};
		}

		TArray<int32> Indices;
		for (int32 i = 0; i < Detections.Num(); ++i)
		{
			Indices.Add(i);
		}

		Indices.Sort([&Detections](int32 A, int32 B)
			{
				return Detections[A].Confidence > Detections[B].Confidence;
			});

		TArray<int32> KeepIndices;

		while (Indices.Num() > 0)
		{
			const int32 CurrentIndex = Indices[0];
			KeepIndices.Add(CurrentIndex);
			Indices.RemoveAt(0);

			for (int32 j = Indices.Num() - 1; j >= 0; --j)
			{
				// class agnostic NMS suppresses boxes of different classes
				/*const float IoU = ComputeIoU(Detections[CurrentIndex], Detections[Indices[j]]);
				if (IoU >= IoUThreshold)
				{
					Indices.RemoveAt(j);
				}*/

				// NMS performed independently per class to avoid suppressions from different classes
				if (Detections[CurrentIndex].ClassID == Detections[Indices[j]].ClassID)
				{
					const float IoU = ComputeIoU(Detections[CurrentIndex], Detections[Indices[j]]);
					if (IoU >= IoUThreshold)
					{
						Indices.RemoveAt(j);
					}
				}
			}
		}

		TArray<FBoundingBox2> Result;
		for (const int32 Index : KeepIndices)
		{
			Result.Add(Detections[Index]);
		}

		return Result;
	}

	TArray<FBoundingBox2> FilterBoxesByClass(const TArray<FBoundingBox2>& Boxes, int32 AllowedClassID)
	{
		TArray<FBoundingBox2> Result;
		for (const FBoundingBox2& Box : Boxes)
		{
			if (Box.ClassID == AllowedClassID)
			{
				Result.Add(Box);
			}
		}
		return Result;
	}

	FFrameRunStats EvaluateFrame(
		int32 FrameIndex,
		const FString& RunId,
		const FString& ScenarioName,
		float TimestampSeconds,
		float InferenceTimeMs,
		float MatchIoUThreshold,
		const TArray<FBoundingBox2>& Detections,
		const TArray<FBoundingBox2>& GroundTruth,
		int32 EvaluatedClassID,
		TArray<int32>* OutDetectionToGT,
		TArray<float>* OutDetectionIoU)
	{
		if (OutDetectionToGT)
		{
			OutDetectionToGT->Init(INDEX_NONE, Detections.Num());
		}
		if (OutDetectionIoU)
		{
			OutDetectionIoU->Init(0.0f, Detections.Num());
		}

		FFrameRunStats Stats;
		Stats.FrameIndex = FrameIndex;
		Stats.RunId = RunId;
		Stats.ScenarioName = ScenarioName;
		Stats.TimestampSeconds = TimestampSeconds;
		Stats.InferenceTimeMs = InferenceTimeMs;
		Stats.NumDetections = Detections.Num();
		Stats.NumGroundTruth = GroundTruth.Num();

		TArray<bool> GTMatched;
		GTMatched.Init(false, GroundTruth.Num());

		float IoUSum = 0.0f;

		for (int32 DetIdx = 0; DetIdx < Detections.Num(); ++DetIdx)
		{
			const FBoundingBox2& Detection = Detections[DetIdx];
			int32 BestGTIndex = INDEX_NONE;
			float BestIoU = 0.0f;

			for (int32 GTIdx = 0; GTIdx < GroundTruth.Num(); ++GTIdx)
			{
				if (GTMatched[GTIdx])
				{
					continue;
				}

				if (Detection.ClassID != GroundTruth[GTIdx].ClassID)
				{
					continue;
				}

				const float IoU = ComputeIoU(Detection, GroundTruth[GTIdx]);
				if (IoU > BestIoU)
				{
					BestIoU = IoU;
					BestGTIndex = GTIdx;
				}
			}

			if (BestGTIndex != INDEX_NONE && BestIoU >= MatchIoUThreshold)
			{
				GTMatched[BestGTIndex] = true;
				Stats.NumTruePositives++;
				IoUSum += BestIoU;

				if (OutDetectionToGT)
				{
					(*OutDetectionToGT)[DetIdx] = BestGTIndex;
				}
				if (OutDetectionIoU)
				{
					(*OutDetectionIoU)[DetIdx] = BestIoU;
				}
			}
			else
			{
				Stats.NumFalsePositives++;
			}
		}

		for (bool bMatched : GTMatched)
		{
			if (!bMatched)
			{
				Stats.NumFalseNegatives++;
			}
		}

		Stats.MeanMatchedIoU = Stats.NumTruePositives > 0
			? IoUSum / Stats.NumTruePositives
			: 0.0f;

		return Stats;
	}
}