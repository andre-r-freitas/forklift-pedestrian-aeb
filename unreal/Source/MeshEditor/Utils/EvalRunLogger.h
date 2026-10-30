#pragma once

#include "CoreMinimal.h"

/**
 * Per-run CSV logger for evaluation runs.
 *
 * Rows are buffered in memory and appended to disk synchronously (every FlushEveryNFrames frames
 * and at EndRun). This replaces the old per-frame background writes, where several thread-pool
 * tasks could append to the same file at the same time and lose or reorder rows.
 *
 * Folder layout:
 *   Saved/EvalRuns/<BatchId>/<RunId>/
 *     detections.csv, ground_truth.csv, frame_metrics.csv, world_state.csv, run_metadata.json
 *     frames/frame_00000.png ...   (clean RGB frames, exactly what the detector saw)
 *     samples/frame_00000.png ...  (annotated failure frames, optional)
 */
class MESHEDITOR_API FEvalRunLogger
{
public:
	void BeginRun(const FString& InBatchId, const FString& InRunId, const FString& InScenarioId, int32 InRepetition);
	void EndRun(const FString& MetadataJson);

	bool IsActive() const { return bActive; }

	const FString& GetRunId() const { return RunId; }
	const FString& GetScenarioId() const { return ScenarioId; }
	int32 GetRepetition() const { return Repetition; }
	const FString& GetRunFolder() const { return RunFolder; }
	FString GetFramesFolder() const { return RunFolder / TEXT("frames"); }
	FString GetSamplesFolder() const { return RunFolder / TEXT("samples"); }

	/** Common prefix for every row: run_id,scenario_id,rep,frame_index,timestamp_s */
	FString RowPrefix(int32 FrameIndex, double TimestampSeconds) const;

	void AddDetectionRow(const FString& Row) { Detections += Row; }
	void AddGroundTruthRow(const FString& Row) { GroundTruth += Row; }
	void AddFrameMetricsRow(const FString& Row) { FrameMetrics += Row; }
	void AddWorldStateRow(const FString& Row) { WorldState += Row; }

	/** Call once per logged frame; flushes to disk periodically. */
	void OnFrameLogged();

	void Flush();

	int32 GetFramesLogged() const { return FramesLogged; }

	int32 FlushEveryNFrames = 25;

private:
	static void AppendToFile(const FString& Path, FString& Buffer);

	bool bActive = false;
	FString BatchId;
	FString RunId;
	FString ScenarioId;
	int32 Repetition = 0;
	FString RunFolder;
	int32 FramesLogged = 0;

	FString Detections;
	FString GroundTruth;
	FString FrameMetrics;
	FString WorldState;
};
