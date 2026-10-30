#pragma once

#include "CoreMinimal.h"
#include "DetectionTypes.generated.h"

USTRUCT(BlueprintType)
struct FBoundingBox2
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    float CenterX = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    float CenterY = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    float Width = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    float Height = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    float Confidence = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    int32 ClassID = -1; // -1 for invalid/no class

    FBoundingBox2() = default;
};
