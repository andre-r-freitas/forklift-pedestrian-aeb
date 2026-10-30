#include "RunIOUtils.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/Paths.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Modules/ModuleManager.h"
#include "UnrealClient.h"

void FAsyncImageSaver::DoWork()
{
	IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));

	TSharedPtr<IImageWrapper> ImageWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);

	ImageWrapper->SetRaw(
		PixelData.GetData(),
		PixelData.Num() * sizeof(FColor),
		Width,
		Height,
		ERGBFormat::BGRA,
		8);

	const TArray64<uint8>& PNGData = ImageWrapper->GetCompressed();
	FFileHelper::SaveArrayToFile(PNGData, *SavePath);
}

void FAsyncDepthSaver::DoWork()
{
	const float MaxCm = FMath::Min(MaxValidCm, 6553.5f);   // 16-bit mm limit: 65.535 m
	TArray<uint16> Millimeters;
	Millimeters.SetNumUninitialized(DepthCm.Num());
	for (int32 i = 0; i < DepthCm.Num(); ++i)
	{
		const float D = DepthCm[i];
		Millimeters[i] = (D > 1.0f && D < MaxCm) ? static_cast<uint16>(FMath::RoundToInt(D * 10.0f)) : 0;
	}

	IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
	TSharedPtr<IImageWrapper> ImageWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);
	ImageWrapper->SetRaw(Millimeters.GetData(), Millimeters.Num() * sizeof(uint16), Width, Height, ERGBFormat::Gray, 16);

	const TArray64<uint8>& PNGData = ImageWrapper->GetCompressed();
	FFileHelper::SaveArrayToFile(PNGData, *SavePath);
}

void FAsyncStringSaver::DoWork()
{
	const uint32 Flags = bAppend ? FILEWRITE_Append : 0;
	FFileHelper::SaveStringToFile(
		Content,
		*SavePath,
		FFileHelper::EEncodingOptions::AutoDetect,
		&IFileManager::Get(),
		Flags);
}

namespace
{
	FORCEINLINE bool IsInsideImage(int32 X, int32 Y, int32 Width, int32 Height)
	{
		return X >= 0 && X < Width && Y >= 0 && Y < Height;
	}

	FORCEINLINE void PutPixel(
		TArray<FColor>& PixelData,
		int32 Width,
		int32 Height,
		int32 X,
		int32 Y,
		const FColor& Color)
	{
		if (!IsInsideImage(X, Y, Width, Height))
		{
			return;
		}

		PixelData[Y * Width + X] = Color;
	}

	void DrawHorizontalLine(
		TArray<FColor>& PixelData,
		int32 Width,
		int32 Height,
		int32 X1,
		int32 X2,
		int32 Y,
		const FColor& Color,
		int32 Thickness)
	{
		if (Width <= 0 || Height <= 0)
		{
			return;
		}

		if (X1 > X2)
		{
			Swap(X1, X2);
		}

		for (int32 Offset = 0; Offset < Thickness; ++Offset)
		{
			const int32 DrawY = Y + Offset;
			if (DrawY < 0 || DrawY >= Height)
			{
				continue;
			}

			const int32 ClampedX1 = FMath::Clamp(X1, 0, Width - 1);
			const int32 ClampedX2 = FMath::Clamp(X2, 0, Width - 1);

			for (int32 X = ClampedX1; X <= ClampedX2; ++X)
			{
				PutPixel(PixelData, Width, Height, X, DrawY, Color);
			}
		}
	}

	void DrawVerticalLine(
		TArray<FColor>& PixelData,
		int32 Width,
		int32 Height,
		int32 X,
		int32 Y1,
		int32 Y2,
		const FColor& Color,
		int32 Thickness)
	{
		if (Width <= 0 || Height <= 0)
		{
			return;
		}

		if (Y1 > Y2)
		{
			Swap(Y1, Y2);
		}

		for (int32 Offset = 0; Offset < Thickness; ++Offset)
		{
			const int32 DrawX = X + Offset;
			if (DrawX < 0 || DrawX >= Width)
			{
				continue;
			}

			const int32 ClampedY1 = FMath::Clamp(Y1, 0, Height - 1);
			const int32 ClampedY2 = FMath::Clamp(Y2, 0, Height - 1);

			for (int32 Y = ClampedY1; Y <= ClampedY2; ++Y)
			{
				PutPixel(PixelData, Width, Height, DrawX, Y, Color);
			}
		}
	}

	void DrawBoundingBoxOutline(
		TArray<FColor>& PixelData,
		int32 ImageWidth,
		int32 ImageHeight,
		const FBoundingBox2& Box,
		const FColor& Color,
		int32 Thickness)
	{
		if (Thickness <= 0)
		{
			return;
		}

		const float HalfWidth = Box.Width * 0.5f;
		const float HalfHeight = Box.Height * 0.5f;

		int32 X1 = FMath::RoundToInt(Box.CenterX - HalfWidth);
		int32 Y1 = FMath::RoundToInt(Box.CenterY - HalfHeight);
		int32 X2 = FMath::RoundToInt(Box.CenterX + HalfWidth);
		int32 Y2 = FMath::RoundToInt(Box.CenterY + HalfHeight);

		if (X1 > X2)
		{
			Swap(X1, X2);
		}

		if (Y1 > Y2)
		{
			Swap(Y1, Y2);
		}

		if (X2 < 0 || Y2 < 0 || X1 >= ImageWidth || Y1 >= ImageHeight)
		{
			return;
		}

		DrawHorizontalLine(PixelData, ImageWidth, ImageHeight, X1, X2, Y1, Color, Thickness);
		DrawHorizontalLine(PixelData, ImageWidth, ImageHeight, X1, X2, Y2, Color, Thickness);
		DrawVerticalLine(PixelData, ImageWidth, ImageHeight, X1, Y1, Y2, Color, Thickness);
		DrawVerticalLine(PixelData, ImageWidth, ImageHeight, X2, Y1, Y2, Color, Thickness);
	}

	void DrawBoundingBoxes(
		TArray<FColor>& PixelData,
		int32 ImageWidth,
		int32 ImageHeight,
		const TArray<FBoundingBox2>& Boxes,
		const FColor& Color,
		int32 Thickness)
	{
		for (const FBoundingBox2& Box : Boxes)
		{
			DrawBoundingBoxOutline(PixelData, ImageWidth, ImageHeight, Box, Color, Thickness);
		}
	}
}

namespace RunIOUtils
{
	FRunOutputPaths InitializeRunOutputFolder(const FString& BaseFolderName)
	{
		FRunOutputPaths Paths;

		Paths.RunId = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
		Paths.RunOutputFolderPath = FPaths::ProjectSavedDir() / BaseFolderName / Paths.RunId;
		Paths.SamplesFolderPath = Paths.RunOutputFolderPath / TEXT("samples");
		Paths.DetectionsCSVPath = Paths.RunOutputFolderPath / TEXT("detections.csv");
		Paths.GroundTruthCSVPath = Paths.RunOutputFolderPath / TEXT("ground_truth.csv");
		Paths.FrameMetricsCSVPath = Paths.RunOutputFolderPath / TEXT("frame_metrics.csv");
		Paths.RunMetadataPath = Paths.RunOutputFolderPath / TEXT("run_metadata.json");

		IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
		PlatformFile.CreateDirectoryTree(*Paths.SamplesFolderPath);

		return Paths;
	}

	void WriteRunMetadata(
		const FString& RunMetadataPath,
		const FString& RunId,
		const FString& ScenarioName,
		int32 ImageWidth,
		int32 ImageHeight,
		float MatchIoUThreshold,
		int32 ImageSaveInterval,
		bool bSaveSampleImages,
		bool bSaveFailureFramesOnly)
	{
		const FString Metadata = FString::Printf(
			TEXT("{\n")
			TEXT("  \"run_id\": \"%s\",\n")
			TEXT("  \"scenario_name\": \"%s\",\n")
			TEXT("  \"image_width\": %d,\n")
			TEXT("  \"image_height\": %d,\n")
			TEXT("  \"match_iou_threshold\": %.3f,\n")
			TEXT("  \"image_save_interval\": %d,\n")
			TEXT("  \"save_sample_images\": %s,\n")
			TEXT("  \"save_failure_frames_only\": %s\n")
			TEXT("}\n"),
			*RunId,
			*ScenarioName,
			ImageWidth,
			ImageHeight,
			MatchIoUThreshold,
			ImageSaveInterval,
			bSaveSampleImages ? TEXT("true") : TEXT("false"),
			bSaveFailureFramesOnly ? TEXT("true") : TEXT("false"));

		FFileHelper::SaveStringToFile(Metadata, *RunMetadataPath);
	}

	void EnsureCSVHeadersWritten(
		const FString& DetectionsCSVPath,
		const FString& GroundTruthCSVPath,
		const FString& FrameMetricsCSVPath)
	{
		FFileHelper::SaveStringToFile(
			TEXT("run_id,frame_index,timestamp_s,class_id,confidence,center_x,center_y,width,height\n"),
			*DetectionsCSVPath);

		FFileHelper::SaveStringToFile(
			TEXT("run_id,frame_index,timestamp_s,class_id,confidence,center_x,center_y,width,height\n"),
			*GroundTruthCSVPath);

		FFileHelper::SaveStringToFile(
			TEXT("run_id,frame_index,timestamp_s,inference_ms,num_detections,num_ground_truth,tp,fp,fn,mean_matched_iou,saved_sample_image\n"),
			*FrameMetricsCSVPath);
	}

	void AppendDetectionsCSV(
		const FString& DetectionsCSVPath,
		const FString& RunId,
		int32 FrameIndex,
		float TimestampSeconds,
		const TArray<FBoundingBox2>& Boxes)
	{
		FString Content;

		for (const FBoundingBox2& Box : Boxes)
		{
			Content += FString::Printf(
				TEXT("%s,%d,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n"),
				*RunId,
				FrameIndex,
				TimestampSeconds,
				Box.ClassID,
				Box.Confidence,
				Box.CenterX,
				Box.CenterY,
				Box.Width,
				Box.Height);
		}

		if (!Content.IsEmpty())
		{
			(new FAutoDeleteAsyncTask<FAsyncStringSaver>(Content, DetectionsCSVPath, true))->StartBackgroundTask();
		}
	}

	void AppendGroundTruthCSV(
		const FString& GroundTruthCSVPath,
		const FString& RunId,
		int32 FrameIndex,
		float TimestampSeconds,
		const TArray<FBoundingBox2>& Boxes)
	{
		FString Content;

		for (const FBoundingBox2& Box : Boxes)
		{
			Content += FString::Printf(
				TEXT("%s,%d,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n"),
				*RunId,
				FrameIndex,
				TimestampSeconds,
				Box.ClassID,
				Box.Confidence,
				Box.CenterX,
				Box.CenterY,
				Box.Width,
				Box.Height);
		}

		if (!Content.IsEmpty())
		{
			(new FAutoDeleteAsyncTask<FAsyncStringSaver>(Content, GroundTruthCSVPath, true))->StartBackgroundTask();
		}
	}

	void AppendFrameMetricsCSV(
		const FString& FrameMetricsCSVPath,
		const FFrameRunStats& Stats)
	{
		const FString Line = FString::Printf(
			TEXT("%s,%d,%.6f,%.3f,%d,%d,%d,%d,%d,%.6f,%s\n"),
			*Stats.RunId,
			Stats.FrameIndex,
			Stats.TimestampSeconds,
			Stats.InferenceTimeMs,
			Stats.NumDetections,
			Stats.NumGroundTruth,
			Stats.NumTruePositives,
			Stats.NumFalsePositives,
			Stats.NumFalseNegatives,
			Stats.MeanMatchedIoU,
			Stats.bSavedSampleImage ? TEXT("true") : TEXT("false"));

		(new FAutoDeleteAsyncTask<FAsyncStringSaver>(Line, FrameMetricsCSVPath, true))->StartBackgroundTask();
	}

	bool ShouldSaveSampleImage(
		const FFrameRunStats& Stats,
		bool bSaveSampleImages,
		bool bSaveFailureFramesOnly,
		int32 ImageSaveInterval)
	{
		if (!bSaveSampleImages)
		{
			return false;
		}

		const bool bFailureFrame =
			(Stats.NumFalsePositives > 0 || Stats.NumFalseNegatives > 0);

		if (bSaveFailureFramesOnly)
		{
			return bFailureFrame;
		}

		return bFailureFrame || (ImageSaveInterval > 0 && Stats.FrameIndex % ImageSaveInterval == 0);
	}

	void ForceOpaqueAlphaIfFirstPixelTransparent(TArray<FColor>& PixelData)
	{
		if (PixelData.IsEmpty() || PixelData[0].A != 0)
		{
			return;
		}

		for (FColor& Pixel : PixelData)
		{
			Pixel.A = 255;
		}
	}

	void QueueRenderTargetForSave(
		UTextureRenderTarget2D* RenderTarget,
		const FString& SavePath)
	{
		if (!RenderTarget)
		{
			return;
		}

		FRenderTarget* RTResource = RenderTarget->GameThread_GetRenderTargetResource();
		if (!RTResource)
		{
			return;
		}

		TArray<FColor> PixelData;
		if (!RTResource->ReadPixels(PixelData))
		{
			return;
		}

		ForceOpaqueAlphaIfFirstPixelTransparent(PixelData);

		(new FAutoDeleteAsyncTask<FAsyncImageSaver>(
			PixelData,
			RenderTarget->SizeX,
			RenderTarget->SizeY,
			SavePath))->StartBackgroundTask();
	}

	void QueueAnnotatedRenderTargetForSave(
		UTextureRenderTarget2D* RenderTarget,
		const FString& SavePath,
		const TArray<FBoundingBox2>& DetectionBoxes,
		const TArray<FBoundingBox2>& GroundTruthBoxes,
		const FColor& DetectionColor,
		const FColor& GroundTruthColor,
		int32 LineThickness)
	{
		if (!RenderTarget)
		{
			return;
		}

		FRenderTarget* RTResource = RenderTarget->GameThread_GetRenderTargetResource();
		if (!RTResource)
		{
			return;
		}

		TArray<FColor> PixelData;
		if (!RTResource->ReadPixels(PixelData))
		{
			return;
		}

		ForceOpaqueAlphaIfFirstPixelTransparent(PixelData);

		const int32 ImageWidth = RenderTarget->SizeX;
		const int32 ImageHeight = RenderTarget->SizeY;

		if (PixelData.Num() != ImageWidth * ImageHeight)
		{
			return;
		}

		const int32 SafeThickness = FMath::Max(1, LineThickness);

		DrawBoundingBoxes(
			PixelData,
			ImageWidth,
			ImageHeight,
			GroundTruthBoxes,
			GroundTruthColor,
			SafeThickness);

		DrawBoundingBoxes(
			PixelData,
			ImageWidth,
			ImageHeight,
			DetectionBoxes,
			DetectionColor,
			SafeThickness);

		(new FAutoDeleteAsyncTask<FAsyncImageSaver>(
			PixelData,
			ImageWidth,
			ImageHeight,
			SavePath))->StartBackgroundTask();
	}

	void QueuePixelDataForSave(
		const TArray<FColor>& PixelData,
		int32 ImageWidth,
		int32 ImageHeight,
		const FString& SavePath)
	{
		if (PixelData.Num() != ImageWidth * ImageHeight || ImageWidth <= 0 || ImageHeight <= 0)
		{
			return;
		}

		TArray<FColor> PixelDataCopy = PixelData;
		ForceOpaqueAlphaIfFirstPixelTransparent(PixelDataCopy);

		(new FAutoDeleteAsyncTask<FAsyncImageSaver>(
			PixelDataCopy,
			ImageWidth,
			ImageHeight,
			SavePath))->StartBackgroundTask();
	}

	void QueueAnnotatedPixelDataForSave(
		const TArray<FColor>& PixelData,
		int32 ImageWidth,
		int32 ImageHeight,
		const FString& SavePath,
		const TArray<FBoundingBox2>& DetectionBoxes,
		const TArray<FBoundingBox2>& GroundTruthBoxes,
		const FColor& DetectionColor,
		const FColor& GroundTruthColor,
		int32 LineThickness)
	{
		if (PixelData.Num() != ImageWidth * ImageHeight || ImageWidth <= 0 || ImageHeight <= 0)
		{
			return;
		}

		TArray<FColor> PixelDataCopy = PixelData;
		ForceOpaqueAlphaIfFirstPixelTransparent(PixelDataCopy);

		const int32 SafeThickness = FMath::Max(1, LineThickness);

		DrawBoundingBoxes(
			PixelDataCopy,
			ImageWidth,
			ImageHeight,
			GroundTruthBoxes,
			GroundTruthColor,
			SafeThickness);

		DrawBoundingBoxes(
			PixelDataCopy,
			ImageWidth,
			ImageHeight,
			DetectionBoxes,
			DetectionColor,
			SafeThickness);

		(new FAutoDeleteAsyncTask<FAsyncImageSaver>(
			PixelDataCopy,
			ImageWidth,
			ImageHeight,
			SavePath))->StartBackgroundTask();
	}

	void QueueDepthForSave(
		const TArray<float>& DepthCm,
		int32 ImageWidth,
		int32 ImageHeight,
		float MaxValidCm,
		const FString& SavePath)
	{
		if (DepthCm.Num() != ImageWidth * ImageHeight || ImageWidth <= 0 || ImageHeight <= 0)
		{
			return;
		}

		(new FAutoDeleteAsyncTask<FAsyncDepthSaver>(DepthCm, ImageWidth, ImageHeight, MaxValidCm, SavePath))->StartBackgroundTask();
	}
}