#include "PedestrianCharacter.h"
#include "AIController.h"
#include "GameFramework/CharacterMovementComponent.h"

// Sets default values
APedestrianCharacter::APedestrianCharacter()
{
    // Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void APedestrianCharacter::BeginPlay()
{
    Super::BeginPlay();

}

// Called every frame
void APedestrianCharacter::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

}

// Called to bind functionality to input
void APedestrianCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
    Super::SetupPlayerInputComponent(PlayerInputComponent);

}

void APedestrianCharacter::ScenarioReset(const FTransform& StartTransform)
{
    // Teleport
    SetActorTransform(StartTransform, false, nullptr, ETeleportType::TeleportPhysics);

    // Stop movement
    if (UCharacterMovementComponent* CharMove = GetCharacterMovement())
    {
        CharMove->StopMovementImmediately();
    }
}

void APedestrianCharacter::ScenarioStart(const TArray<FVector>& Waypoints, float Speed) {

    // FIX: make sure the configured speed is actually used by AI MoveTo (which moves the
    // character at MaxWalkSpeed). If the Blueprint sets MaxWalkSpeed itself, it overrides this.
    if (UCharacterMovementComponent* CharMove = GetCharacterMovement())
    {
        if (Speed > 0.0f)
        {
            CharMove->MaxWalkSpeed = Speed;
        }
    }

    ReceiveScenarioStart(Waypoints, Speed);
}

void APedestrianCharacter::ScenarioSetCrouched(bool bCrouched)
{
    bScenarioCrouched = bCrouched;
}

void APedestrianCharacter::ScenarioStop()
{
    if (UCharacterMovementComponent* CharMove = GetCharacterMovement())
    {
        CharMove->StopMovementImmediately();
    }
}