#include "Utils/EvalRunLogger.h"

#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

void FEvalRunLogger::BeginRun(const FString& InBatchId, const FString& InRunId, const FString& InScenarioId, int32 InRepetition)
{
	if (bActive)
	{
		EndRun(TEXT("{ \"note\": \"run was not ended explicitly\" }\n"));
	}

	BatchId = InBatchId;
	RunId = InRunId;
	ScenarioId = InScenarioId;
	Repetition = InRepetition;
	FramesLogged = 0;

	RunFolder = FPaths::ProjectSavedDir() / TEXT("EvalRuns") / BatchId / RunId;
	IFileManager::Get().MakeDirectory(*(RunFolder / TEXT("frames")), true);
	IFileManager::Get().MakeDirectory(*(RunFolder / TEXT("samples")), true);

	const FString Prefix = TEXT("run_id,scenario_id,rep,frame_index,timestamp_s,");

	Detections = Prefix + TEXT("class_id,confidence,center_x,center_y,width,height,matched_stencil_id,matched_iou,depth_m,est_x_m,est_y_m,fwd_m,lat_m,in_zone\n");
	GroundTruth = Prefix + TEXT("stencil_id,actor_name,class_id,center_x,center_y,width,height,visible_px,full_px,detected\n");
	FrameMetrics = Prefix + TEXT("inference_ms,num_detections,num_ground_truth,tp,fp,fn,mean_matched_iou,zone_hit,brake_state,trigger_dist_m,stale_frame\n");
	WorldState = Prefix + TEXT("role,actor_name,stencil_id,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,yaw_deg,pitch_deg\n");

	bActive = true;
	Flush(); // writes the headers
}

void FEvalRunLogger::EndRun(const FString& MetadataJson)
{
	if (!bActive)
	{
		return;
	}

	Flush();
	FFileHelper::SaveStringToFile(MetadataJson, *(RunFolder / TEXT("run_metadata.json")));
	bActive = false;
}

FString FEvalRunLogger::RowPrefix(int32 FrameIndex, double TimestampSeconds) const
{
	return FString::Printf(TEXT("%s,%s,%d,%d,%.4f,"), *RunId, *ScenarioId, Repetition, FrameIndex, TimestampSeconds);
}

void FEvalRunLogger::OnFrameLogged()
{
	++FramesLogged;
	if (FlushEveryNFrames > 0 && FramesLogged % FlushEveryNFrames == 0)
	{
		Flush();
	}
}

void FEvalRunLogger::Flush()
{
	if (RunFolder.IsEmpty())
	{
		return;
	}

	AppendToFile(RunFolder / TEXT("detections.csv"), Detections);
	AppendToFile(RunFolder / TEXT("ground_truth.csv"), GroundTruth);
	AppendToFile(RunFolder / TEXT("frame_metrics.csv"), FrameMetrics);
	AppendToFile(RunFolder / TEXT("world_state.csv"), WorldState);
}

void FEvalRunLogger::AppendToFile(const FString& Path, FString& Buffer)
{
	if (Buffer.IsEmpty())
	{
		return;
	}

	FFileHelper::SaveStringToFile(Buffer, *Path, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM,
		&IFileManager::Get(), FILEWRITE_Append);
	Buffer.Reset();
}
