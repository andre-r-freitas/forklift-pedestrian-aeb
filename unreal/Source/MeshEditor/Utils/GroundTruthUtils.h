#pragma once

#include "CoreMinimal.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Types/GroundTruthTypes.h"

class AActor;
class UWorld;

namespace GroundTruthUtils
{
	FGroundTruthAssignmentResult AssignStencilIDsToTaggedActors(UWorld* World, const FName& ActorTag, int32 DefaultClassID);

	// R channel = visible (depth-tested) stencil ID. If bReadFullSilhouetteFromG is true, the G channel is read
	// as the full-silhouette stencil ID (no depth test) to compute FullPixelCounts; otherwise those are -1.
	FGroundTruthExtractionResult ExtractBoxesFromStencilRenderTarget(
		UTextureRenderTarget2D* RenderTarget,
		const TMap<int32, int32>& StencilIDToClassID,
		bool bReadFullSilhouetteFromG = false,
		TArray<FColor>* OutPixels = nullptr);   // if set, receives the raw stencil image (R = visible, G = full)

	void ApplyStencilIDToActor(AActor* Actor, int32 StencilID);
}
