using UnrealBuildTool;
using System.Collections.Generic;

public class MeshEditorTarget : TargetRules
{
	public MeshEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V5;

		ExtraModuleNames.AddRange( new string[] { "MeshEditor" } );
	}
}
