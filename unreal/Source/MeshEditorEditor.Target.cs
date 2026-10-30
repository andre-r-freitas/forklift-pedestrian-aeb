using UnrealBuildTool;
using System.Collections.Generic;

public class MeshEditorEditorTarget : TargetRules
{
	public MeshEditorEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.V5;

		ExtraModuleNames.AddRange( new string[] { "MeshEditor" } );
	}
}
