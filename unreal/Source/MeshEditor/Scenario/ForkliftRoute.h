#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SplineComponent.h"
#include "ForkliftRoute.generated.h"

UCLASS()
class MESHEDITOR_API AForkliftRoute : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AForkliftRoute();

	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Route")
	TObjectPtr<USplineComponent> Spline;

	UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Route")
	FName RouteId;

	UPROPERTY(EditInstanceOnly, BlueprintReadOnly, Category = "Route")
	float SpeedKmh = 3.0f;

	UFUNCTION(BlueprintCallable, Category = "Route")
	USplineComponent* GetSpline() const { return Spline; }

	UFUNCTION(BlueprintCallable, Category = "Route")
	FVector GetStartLocation() const;

	UFUNCTION(BlueprintCallable, Category = "Route")
	FRotator GetStartRotation() const;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
};
