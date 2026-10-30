#include "Scenario/ForkliftRoute.h"

// Sets default values
AForkliftRoute::AForkliftRoute()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	Spline = CreateDefaultSubobject<USplineComponent>(TEXT("Spline"));
	RootComponent = Spline;

}

// Called when the game starts or when spawned
void AForkliftRoute::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AForkliftRoute::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

FVector AForkliftRoute::GetStartLocation() const
{
    if (!Spline || Spline->GetNumberOfSplinePoints() == 0)
    {
        return GetActorLocation();
    }

    return Spline->GetLocationAtSplinePoint(
        0,
        ESplineCoordinateSpace::World
    );
}

FRotator AForkliftRoute::GetStartRotation() const
{
    if (!Spline || Spline->GetNumberOfSplinePoints() == 0)
    {
        return GetActorRotation();
    }

    return Spline->GetRotationAtSplinePoint(
        0,
        ESplineCoordinateSpace::World
    );
}

