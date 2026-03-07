; ═══════════════════════════════════════════════════════════════════
;  theSpaceDB  —  Inno Setup Script
;  Compiles to a proper Windows installer wizard (.exe)
;
;  To compile:
;    1. Download Inno Setup from https://jrsoftware.org/isinfo.php
;    2. Open this file in Inno Setup Compiler
;    3. Press Ctrl+F9 to compile
;    4. Installer .exe will appear in installer/Output/
; ═══════════════════════════════════════════════════════════════════

#define AppName      "theSpaceDB"
#define AppVersion   "0.1.0"
#define AppPublisher "Draeghir"
#define AppURL       "https://github.com/nihar840/theSpaceDB"
#define AppExeName   "spacesh.bat"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}/releases
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputDir=Output
OutputBaseFilename=theSpaceDB-{#AppVersion}-setup
SetupIconFile=
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardSizePercent=130
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\spacesh.bat
ChangesEnvironment=yes

; ── wizard appearance ────────────────────────────────────────────────
WizardImageFile=compiler:WizClassicImage.bmp
WizardSmallImageFile=compiler:WizClassicSmallImage.bmp

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
english.WelcomeLabel2=This will install [name/ver] on your computer.%n%ntheSpaceDB is a self-evolving cognitive database.%nBlocks float in free space. Distance evolves.%nClusters emerge. Personalities form.%n%nNothing fixed. Everything evolving. We chase infinity. ∞

[Tasks]
Name: "desktopicon";   Description: "Create a &desktop shortcut";  GroupDescription: "Additional icons:"; Flags: checked
Name: "addtopath";     Description: "Add spacesh to system &PATH"; GroupDescription: "Additional icons:"; Flags: checked

[Files]
; The installer scripts — we run them from here
Source: "install.ps1";   DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName} Shell";        Filename: "cmd.exe"; Parameters: "/k ""{app}\spacesh.bat"""; WorkingDir: "{userdocs}"
Name: "{group}\Uninstall {#AppName}";    Filename: "{uninstallexe}"
Name: "{commondesktop}\spacesh";         Filename: "cmd.exe"; Parameters: "/k ""{app}\spacesh.bat"""; WorkingDir: "{userdocs}"; Tasks: desktopicon

[Run]
; Run PowerShell installer in unattended mode after wizard finishes
Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"" -InstallDir ""{app}"" -Unattended"; \
    StatusMsg: "Installing theSpaceDB and dependencies (this may take a few minutes)..."; \
    Flags: runhidden waituntilterminated

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}"

[Code]

// ── Check Python before proceeding ──────────────────────────────────
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
  PythonOK: Boolean;
begin
  PythonOK := False;

  // Check for python / python3
  if Exec('python', '--version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    PythonOK := (ResultCode = 0);

  if not PythonOK then
    if Exec('python3', '--version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      PythonOK := (ResultCode = 0);

  if not PythonOK then
  begin
    if MsgBox(
      'Python 3.11 or higher is required.' + #13#10 +
      #13#10 +
      'The installer will attempt to install Python automatically via winget.' + #13#10 +
      'Click Yes to continue, or No to install Python manually from python.org.',
      mbConfirmation, MB_YESNO
    ) = IDNO then
    begin
      Result := False;
      Exit;
    end;
  end;

  Result := True;
end;

// ── Add to PATH ──────────────────────────────────────────────────────
procedure CurStepChanged(CurStep: TSetupStep);
var
  AppPath, OldPath, NewPath: String;
begin
  if (CurStep = ssPostInstall) and IsTaskSelected('addtopath') then
  begin
    AppPath := ExpandConstant('{app}');
    OldPath := GetEnvironmentVariable('PATH');
    if Pos(AppPath, OldPath) = 0 then
    begin
      NewPath := OldPath + ';' + AppPath;
      SetEnvironmentVariable('PATH', NewPath);
      RegWriteStringValue(
        HKEY_CURRENT_USER,
        'Environment',
        'PATH',
        NewPath
      );
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppPath, OldPath, NewPath: String;
  Paths: TArrayOfString;
  I: Integer;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    AppPath := ExpandConstant('{app}');
    OldPath := GetEnvironmentVariable('PATH');
    // Remove our entry from PATH
    StringChangeEx(OldPath, ';' + AppPath, '', True);
    StringChangeEx(OldPath, AppPath + ';', '', True);
    RegWriteStringValue(HKEY_CURRENT_USER, 'Environment', 'PATH', OldPath);
  end;
end;
