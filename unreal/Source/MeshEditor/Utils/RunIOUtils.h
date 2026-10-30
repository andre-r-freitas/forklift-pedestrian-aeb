#pragma once

#include "CoreMinimal.h"
#include "Async/AsyncWork.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Types/DetectionTypes.h"
#include "Types/EvaluationTypes.h"

class FAsyncImageSaver : public FNonAbandonableTask
{
	friend class FAutoDeleteAsyncTask<FAsyncImageSaver>;

	TArray<FColor> PixelData;
	int32 Width = 0;
	int32 Height = 0;
	FString SavePath;

public:
	FAsyncImageSaver(const TArray<FColor>& InData, int32 InWidth, int32 InHeight, const FString& InPath)
		: PixelData(InData), Width(InWidth), Height(InHeight), SavePath(InPath)
	{
	}

	void DoWork();

	FORCEINLINE TStatId GetStatId() const
	{
		RETURN_QUICK_DECLARE_CYCLE_STAT(FAsyncImageSaver, STATGROUP_ThreadPoolAsyncTasks);
	}
};

/** Saves planar depth (cm, float) as a 16-bit grayscale PNG in millimeters; 0 = invalid / beyond 65.535 m. */
class FAsyncDepthSaver : public FNonAbandonableTask
{
	friend class FAutoDeleteAsyncTask<FAsyncDepthSaver>;

	TArray<float> DepthCm;
	int32 Width = 0;
	int32 Height = 0;
	float MaxValidCm = 0.0f;
	FString SavePath;

public:
	FAsyncDepthSaver(const TArray<float>& InDepthCm, int32 InWidth, int32 InHeight, float InMaxValidCm, const FString& InPath)
		: DepthCm(InDepthCm), Width(InWidth), Height(InHeight), MaxValidCm(InMaxValidCm), SavePath(InPath)
	{
	}

	void DoWork();

	FORCEINLINE TStatId GetStatId() const
	{
		RETURN_QUICK_DECLARE_CYCLE_STAT(FAsyncDepthSaver, STATGROUP_ThreadPoolAsyncTasks);
	}
};

class FAsyncStringSaver : public FNonAbandonableTask
{
	friend class FAutoDeleteAsyncTask<FAsyncStringSaver>;

	FString Content;
	FString SavePath;
	bool bAppend = false;

public:
	FAsyncStringSaver(const FString& InContent, const FString& InSavePath, bool bInAppend)
		: Content(InContent), SavePath(InSavePath), bAppend(bInAppend)
	{
	}

	void DoWork();

	FORCEINLINE TStatId GetStatId() const
	{
		RETURN_QUICK_DECLARE_CYCLE_STAT(FAsyncStringSaver, STATGROUP_ThreadPoolAsyncTasks);
	}
};

namespace RunIOUtils
{
	struct FRunOutputPaths
	{
		FString RunId;
		FString RunOutputFolderPath;
		FString SamplesFolderPath;
		FString DetectionsCSVPath;
		FString GroundTruthCSVPath;
		FString FrameMetricsCSVPath;
		FString RunMetadataPath;
	};

	FRunOutputPaths InitializeRunOutputFolder(const FString& BaseFolderName = TEXT("YOLORuns"));

	void WriteRunMetadata(
		const FString& RunMetadataPath,
		const FString& RunId,
		const FString& ScenarioName,
		int32 ImageWidth,
		int32 ImageHeight,
		float MatchIoUThreshold,
		int32 ImageSaveInterval,
		bool bSaveSampleImages,
		bool bSaveFailureFramesOnly);

	void EnsureCSVHeadersWritten(
		const FString& DetectionsCSVPath,
		const FString& GroundTruthCSVPath,
		const FString& FrameMetricsCSVPath);

	void AppendDetectionsCSV(
		const FString& DetectionsCSVPath,
		const FString& RunId,
		int32 FrameIndex,
		float TimestampSeconds,
		const TArray<FBoundingBox2>& Boxes);

	void AppendGroundTruthCSV(
		const FString& GroundTruthCSVPath,
		const FString& RunId,
		int32 FrameIndex,
		float TimestampSeconds,
		const TArray<FBoundingBox2>& Boxes);

	void AppendFrameMetricsCSV(
		const FString& FrameMetricsCSVPath,
		const FFrameRunStats& Stats);

	bool ShouldSaveSampleImage(
		const FFrameRunStats& Stats,
		bool bSaveSampleImages,
		bool bSaveFailureFramesOnly,
		int32 ImageSaveInterval);

	void ForceOpaqueAlphaIfFirstPixelTransparent(TArray<FColor>& PixelData);

	void QueueRenderTargetForSave(
		UTextureRenderTarget2D* RenderTarget,
		const FString& SavePath);

	void QueueAnnotatedRenderTargetForSave(
		UTextureRenderTarget2D* RenderTarget,
		const FString& SavePath,
		const TArray<FBoundingBox2>& DetectionBoxes,
		const TArray<FBoundingBox2>& GroundTruthBoxes,
		const FColor& DetectionColor = FColor(0, 255, 0, 255),
		const FColor& GroundTruthColor = FColor(255, 0, 0, 255),
		int32 LineThickness = 2);

	void QueuePixelDataForSave(
		const TArray<FColor>& PixelData,
		int32 ImageWidth,
		int32 ImageHeight,
		const FString& SavePath);

	/** Depth frame as 16-bit PNG (mm). Values <= 1 cm or >= MaxValidCm are written as 0. Background thread. */
	void QueueDepthForSave(
		const TArray<float>& DepthCm,
		int32 ImageWidth,
		int32 ImageHeight,
		float MaxValidCm,
		const FString& SavePath);

	void QueueAnnotatedPixelDataForSave(
		const TArray<FColor>& PixelData,
		int32 ImageWidth,
		int32 ImageHeight,
		const FString& SavePath,
		const TArray<FBoundingBox2>& DetectionBoxes,
		const TArray<FBoundingBox2>& GroundTruthBoxes,
		const FColor& DetectionColor = FColor::Red,
		const FColor& GroundTruthColor = FColor::Green,
		int32 LineThickness = 2);
}

