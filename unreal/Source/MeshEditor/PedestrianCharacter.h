#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "AIController.h"
#include "PedestrianCharacter.generated.h"

UCLASS()
class MESHEDITOR_API APedestrianCharacter : public ACharacter
{
	GENERATED_BODY()

public:
	// Sets default values for this character's properties
	APedestrianCharacter();

	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

	UFUNCTION(BlueprintCallable, Category = "Scenario")
	void ScenarioReset(const FTransform& StartTransform);

	UFUNCTION(BlueprintCallable, Category = "Scenario")
	void ScenarioStart(const TArray<FVector>& Waypoints, float Speed);

	UFUNCTION(BlueprintCallable, Category = "Scenario")
	void ScenarioStop();

	/** Called by the ScenarioManager at every reset (true or false), before the run starts. */
	UFUNCTION(BlueprintCallable, Category = "Scenario")
	void ScenarioSetCrouched(bool bCrouched);

	/** Read this in the Animation Blueprint (Try Get Pawn Owner -> Cast -> Scenario Crouched) to select the crouch pose. */
	UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "Scenario")
	bool bScenarioCrouched = false;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	UFUNCTION(BlueprintImplementableEvent, Category = "Scenario")
	void ReceiveScenarioReset(const FTransform& StartTransform);

	UFUNCTION(BlueprintImplementableEvent, Category = "Scenario")
	void ReceiveScenarioStart(const TArray<FVector>& Waypoints, float Speed);

	UFUNCTION(BlueprintImplementableEvent, Category = "Scenario")
	void ReceiveScenarioStop();
};

