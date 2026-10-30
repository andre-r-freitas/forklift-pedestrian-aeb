#include "Utils/GroundTruthUtils.h"

#include "Components/PrimitiveComponent.h"
#include "EngineUtils.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/TextureRenderTarget.h"
#include "GameFramework/Actor.h"

namespace
{
	struct FPixelBoundsAccumulator
	{
		int32 MinX = TNumericLimits<int32>::Max();
		int32 MinY = TNumericLimits<int32>::Max();
		int32 MaxX = TNumericLimits<int32>::Lowest();
		int32 MaxY = TNumericLimits<int32>::Lowest();
		int32 PixelCount = 0;

		void AddPixel(int32 X, int32 Y)
		{
			MinX = FMath::Min(MinX, X);
			MinY = FMath::Min(MinY, Y);
			MaxX = FMath::Max(MaxX, X);
			MaxY = FMath::Max(MaxY, Y);
			++PixelCount;
		}

		bool IsValid() const
		{
			return PixelCount > 0 && MinX <= MaxX && MinY <= MaxY;
		}
	};
}

namespace GroundTruthUtils
{
	void ApplyStencilIDToActor(AActor* Actor, int32 StencilID)
	{
		if (!IsValid(Actor))
		{
			return;
		}

		TArray<UPrimitiveComponent*> PrimitiveComponents;
		Actor->GetComponents<UPrimitiveComponent>(PrimitiveComponents);

		int32 NumApplied = 0;

		for (UPrimitiveComponent* PrimitiveComp : PrimitiveComponents)
		{
			if (!IsValid(PrimitiveComp))
			{
				continue;
			}

			if (!PrimitiveComp->IsRegistered())
			{
				continue;
			}

			PrimitiveComp->SetRenderCustomDepth(true);
			PrimitiveComp->SetCustomDepthStencilValue(StencilID);
			PrimitiveComp->MarkRenderStateDirty();
			++NumApplied;
		}

		UE_LOG(LogTemp, Log, TEXT("GroundTruthUtils: Applied stencil %d to actor '%s' on %d primitive components."),
			StencilID,
			*Actor->GetName(),
			NumApplied);
	}

	FGroundTruthAssignmentResult AssignStencilIDsToTaggedActors(UWorld* World, const FName& ActorTag, int32 DefaultClassID)
	{
		FGroundTruthAssignmentResult Result;

		if (!World)
		{
			UE_LOG(LogTemp, Warning, TEXT("GroundTruthUtils: AssignStencilIDsToTaggedActors failed because World is null."));
			return Result;
		}

		TArray<AActor*> TaggedActors;
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* Actor = *It;
			if (IsValid(Actor) && Actor->ActorHasTag(ActorTag))
			{
				TaggedActors.Add(Actor);
			}
		}

		TaggedActors.Sort([](const AActor& A, const AActor& B)
			{
				return A.GetName() < B.GetName();
			});

		if (TaggedActors.Num() == 0)
		{
			UE_LOG(LogTemp, Warning, TEXT("GroundTruthUtils: No actors found with tag '%s'."), *ActorTag.ToString());
			return Result;
		}

		constexpr int32 MinStencilID = 1;
		constexpr int32 MaxStencilID = 255;

		if (TaggedActors.Num() > MaxStencilID)
		{
			UE_LOG(LogTemp, Error, TEXT("GroundTruthUtils: Found %d tagged actors, but only %d stencil IDs are available."),
				TaggedActors.Num(),
				MaxStencilID);
			return Result;
		}

		int32 NextStencilID = MinStencilID;

		for (AActor* Actor : TaggedActors)
		{
			if (!IsValid(Actor))
			{
				continue;
			}

			const int32 AssignedStencilID = NextStencilID++;
			ApplyStencilIDToActor(Actor, AssignedStencilID);

			FGroundTruthInstanceInfo InstanceInfo;
			InstanceInfo.StencilID = AssignedStencilID;
			InstanceInfo.ActorName = FName(*Actor->GetName());
			InstanceInfo.ClassID = DefaultClassID;

			Result.Instances.Add(InstanceInfo);
			Result.StencilIDToClassID.Add(AssignedStencilID, DefaultClassID);
			Result.StencilIDToActorName.Add(AssignedStencilID, InstanceInfo.ActorName);
			Result.StencilIDToActor.Add(AssignedStencilID, Actor);

			UE_LOG(LogTemp, Log, TEXT("GroundTruthUtils: Actor '%s' assigned stencil ID %d with ClassID %d."),
				*Actor->GetName(),
				AssignedStencilID,
				DefaultClassID);
		}

		UE_LOG(LogTemp, Log, TEXT("GroundTruthUtils: Assigned stencil IDs to %d tagged actors."),
			Result.Instances.Num());

		return Result;
	}

	FGroundTruthExtractionResult ExtractBoxesFromStencilRenderTarget(
		UTextureRenderTarget2D* RenderTarget,
		const TMap<int32, int32>& StencilIDToClassID,
		bool bReadFullSilhouetteFromG,
		TArray<FColor>* OutPixels)
	{
		FGroundTruthExtractionResult Result;

		if (!RenderTarget)
		{
			UE_LOG(LogTemp, Warning, TEXT("GroundTruthUtils: ExtractBoxesFromStencilRenderTarget failed because RenderTarget is null."));
			return Result;
		}

		FTextureRenderTargetResource* RTResource = RenderTarget->GameThread_GetRenderTargetResource();
		if (!RTResource)
		{
			UE_LOG(LogTemp, Warning, TEXT("GroundTruthUtils: Render target resource is null."));
			return Result;
		}

		TArray<FColor> Pixels;
		if (!RTResource->ReadPixels(Pixels))
		{
			UE_LOG(LogTemp, Warning, TEXT("GroundTruthUtils: ReadPixels failed."));
			return Result;
		}

		const int32 Width = RenderTarget->SizeX;
		const int32 Height = RenderTarget->SizeY;

		if (Pixels.Num() != Width * Height)
		{
			UE_LOG(LogTemp, Warning, TEXT("GroundTruthUtils: Pixel count mismatch. Got %d, expected %d."),
				Pixels.Num(),
				Width * Height);
			return Result;
		}

		if (OutPixels)
		{
			*OutPixels = Pixels;
		}

		TMap<int32, FPixelBoundsAccumulator> BoundsByStencilID;
		TMap<int32, int32> FullPixelsByStencilID;

		for (int32 Y = 0; Y < Height; ++Y)
		{
			for (int32 X = 0; X < Width; ++X)
			{
				const int32 PixelIndex = Y * Width + X;
				const FColor& Pixel = Pixels[PixelIndex];

				if (bReadFullSilhouetteFromG && Pixel.G > 0)
				{
					FullPixelsByStencilID.FindOrAdd(static_cast<int32>(Pixel.G))++;
				}

				// Assumption: visible instance/stencil ID is encoded in the R channel.
				const int32 StencilID = static_cast<int32>(Pixel.R);

				if (StencilID <= 0)
				{
					continue;
				}

				FPixelBoundsAccumulator& Accumulator = BoundsByStencilID.FindOrAdd(StencilID);
				Accumulator.AddPixel(X, Y);
			}
		}

		for (const TPair<int32, FPixelBoundsAccumulator>& Pair : BoundsByStencilID)
		{
			UE_LOG(LogTemp, Verbose, TEXT("Observed in RT: Stencil %d, PixelCount %d"),
				Pair.Key, Pair.Value.PixelCount);
		}

		for (const TPair<int32, FPixelBoundsAccumulator>& Pair : BoundsByStencilID)
		{
			const int32 StencilID = Pair.Key;
			const FPixelBoundsAccumulator& Bounds = Pair.Value;

			if (!Bounds.IsValid())
			{
				continue;
			}

			const int32* ClassIDPtr = StencilIDToClassID.Find(StencilID);
			if (!ClassIDPtr)
			{
				UE_LOG(LogTemp, Warning, TEXT("GroundTruthUtils: Found visible stencil ID %d in GT render target, but no class mapping exists for it."),
					StencilID);
				continue;
			}

			FBoundingBox2 Box;
			Box.CenterX = (static_cast<float>(Bounds.MinX + Bounds.MaxX) * 0.5f);
			Box.CenterY = (static_cast<float>(Bounds.MinY + Bounds.MaxY) * 0.5f);
			Box.Width = static_cast<float>(Bounds.MaxX - Bounds.MinX + 1);
			Box.Height = static_cast<float>(Bounds.MaxY - Bounds.MinY + 1);
			Box.ClassID = *ClassIDPtr;
			Box.Confidence = 1.0f;

			Result.Boxes.Add(Box);
			Result.ObservedStencilIDs.Add(StencilID);
			Result.BoxStencilIDs.Add(StencilID);
			Result.VisiblePixelCounts.Add(Bounds.PixelCount);

			const int32* FullCount = FullPixelsByStencilID.Find(StencilID);
			Result.FullPixelCounts.Add(bReadFullSilhouetteFromG ? (FullCount ? *FullCount : 0) : -1);

			UE_LOG(LogTemp, Log, TEXT("GroundTruthUtils: Extracted GT box for stencil %d -> Center(%.1f, %.1f), Size(%.1f, %.1f), ClassID=%d"),
				StencilID,
				Box.CenterX,
				Box.CenterY,
				Box.Width,
				Box.Height,
				Box.ClassID);
		}

		Result.ObservedStencilIDs.Sort();

		UE_LOG(LogTemp, Log, TEXT("GroundTruthUtils: Extracted %d GT boxes from render target."), Result.Boxes.Num());

		return Result;
	}
}