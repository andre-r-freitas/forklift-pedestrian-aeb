using UnrealBuildTool;

public class MeshEditor : ModuleRules
{
	public MeshEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "EnhancedInput",
            "AIModule",
            "StateTreeModule",
            "GameplayStateTreeModule",
            "UMG",
            "NNE",
            "ImageWrapper",
            "RenderCore",
            "ChaosVehicles"
        });

        PrivateDependencyModuleNames.AddRange(new string[] {  });

        PublicIncludePaths.AddRange(new string[] {
            "MeshEditor",
            "MeshEditor/Scenario"
        });
    }
}
