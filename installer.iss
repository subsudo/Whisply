[Setup]
AppName=Whisply
AppVersion=0.1.2
AppVerName=Whisply 0.1.2
VersionInfoVersion=0.1.2.0
DefaultDirName={localappdata}\Programs\Whisply
DefaultGroupName=Whisply
OutputDir=dist
OutputBaseFilename=Whisply-Installer
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\Whisply.exe
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
CloseApplications=yes
CloseApplicationsFilter=Whisply.exe
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"

[CustomMessages]
TaskGroup=Additional tasks:
TaskDesktopIcon=Create a desktop shortcut
TaskStartMenuIcon=Create Start menu entries
TaskAutostart=Start with Windows
RunWhisply=Start Whisply
UninstallTitle=Uninstall Whisply
UninstallPrompt=Choose which Whisply data should also be removed.
UninstallConfig=Delete configuration
UninstallLogs=Delete logs
UninstallModels=Delete models
UninstallCudaRuntime=Delete Whisply CUDA runtime (downloaded by Whisply)
UninstallAppRunning=Whisply is still running. Close it to continue uninstall.
UninstallAppRunningManual=Whisply could not be closed automatically. Please close it from the tray and click Retry.
BtnCancel=Cancel

german.TaskGroup=Zusätzliche Aufgaben:
german.TaskDesktopIcon=Desktop-Verknüpfung erstellen
german.TaskStartMenuIcon=Startmenü-Einträge erstellen
german.TaskAutostart=Mit Windows starten
german.RunWhisply=Whisply starten
german.UninstallTitle=Whisply deinstallieren
german.UninstallPrompt=Wähle aus, welche Whisply-Daten zusätzlich gelöscht werden sollen.
german.UninstallConfig=Konfiguration löschen
german.UninstallLogs=Logs löschen
german.UninstallModels=Modelle löschen
german.UninstallCudaRuntime=Whisply CUDA-Runtime löschen (von Whisply heruntergeladen)
german.UninstallAppRunning=Whisply läuft noch. Bitte zuerst schließen, um die Deinstallation fortzusetzen.
german.UninstallAppRunningManual=Whisply konnte nicht automatisch geschlossen werden. Bitte im Tray beenden und dann Wiederholen klicken.
german.BtnCancel=Abbrechen

[Files]
Source: "dist\Whisply.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "config.yaml"; DestDir: "{app}"; Flags: ignoreversion
Source: "cuda_manifest.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\icon.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "assets\icon.png"; DestDir: "{app}\assets"; Flags: ignoreversion

[Icons]
Name: "{group}\Whisply"; Filename: "{app}\Whisply.exe"; IconFilename: "{app}\assets\icon.ico"; Tasks: startmenuicon
Name: "{group}\Uninstall Whisply"; Filename: "{uninstallexe}"; Tasks: startmenuicon
Name: "{userdesktop}\Whisply"; Filename: "{app}\Whisply.exe"; IconFilename: "{app}\assets\icon.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:TaskDesktopIcon}"; GroupDescription: "{cm:TaskGroup}"; Flags: checkedonce
Name: "startmenuicon"; Description: "{cm:TaskStartMenuIcon}"; GroupDescription: "{cm:TaskGroup}"; Flags: checkedonce
Name: "autostart"; Description: "{cm:TaskAutostart}"; GroupDescription: "{cm:TaskGroup}"; Flags: checkedonce

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "Whisply"; ValueData: "{app}\Whisply.exe"; Tasks: autostart; Flags: uninsdeletevalue

[Run]
Filename: "{app}\Whisply.exe"; Description: "{cm:RunWhisply}"; Flags: nowait postinstall skipifsilent

[Code]
var
  RemoveConfig: Boolean;
  RemoveLogs: Boolean;
  RemoveModels: Boolean;
  RemoveCudaRuntime: Boolean;

function SelectedUiLanguageCode(): string;
begin
  if ActiveLanguage = 'german' then
    Result := 'de'
  else
    Result := 'en';
end;

procedure EnsureInitialUiLanguageConfig();
var
  ConfigPath: string;
  Content: string;
begin
  ConfigPath := ExpandConstant('{userappdata}\Whisply\config.yaml');
  if FileExists(ConfigPath) then
    exit;
  if not ForceDirectories(ExtractFileDir(ConfigPath)) then
    exit;
  Content := 'general:' + #13#10 +
             '  language_ui: ' + SelectedUiLanguageCode() + #13#10 +
             'whisper:' + #13#10 +
             '  language: ' + SelectedUiLanguageCode() + #13#10;
  if not SaveStringToFile(ConfigPath, Content, False) then
  begin
    MsgBox('Could not write initial config: ' + ConfigPath, mbError, MB_OK);
    exit;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    EnsureInitialUiLanguageConfig();
  end;
end;

function IsWhisplyRunning(): Boolean;
var
  ResultCode: Integer;
  CmdLine: string;
begin
  Result := False;
  CmdLine := '/C tasklist /FI "IMAGENAME eq Whisply.exe" /NH | find /I "Whisply.exe" >NUL';
  if not Exec(ExpandConstant('{cmd}'), CmdLine, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    exit;
  Result := (ResultCode = 0);
end;

function TryStopWhisply(): Boolean;
var
  ResultCode: Integer;
begin
  Exec(
    ExpandConstant('{cmd}'),
    '/C taskkill /IM Whisply.exe /T /F >NUL 2>&1',
    '',
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  );
  Result := not IsWhisplyRunning();
end;

function EnsureWhisplyClosed(): Boolean;
var
  UserChoice: Integer;
begin
  Result := True;
  if not IsWhisplyRunning() then
    exit;

  if UninstallSilent then
  begin
    Result := TryStopWhisply();
    exit;
  end;

  if TryStopWhisply() then
    exit;

  while IsWhisplyRunning() do
  begin
    UserChoice := SuppressibleMsgBox(
      ExpandConstant('{cm:UninstallAppRunning}'#13#10#13#10'{cm:UninstallAppRunningManual}'),
      mbError,
      MB_RETRYCANCEL,
      IDRETRY
    );
    if UserChoice <> IDRETRY then
    begin
      Result := False;
      exit;
    end;
    TryStopWhisply();
  end;
end;

function ShowUninstallDataDialog(): Boolean;
var
  Form: TSetupForm;
  HeaderLabel: TNewStaticText;
  ConfigCheck: TNewCheckBox;
  LogsCheck: TNewCheckBox;
  ModelsCheck: TNewCheckBox;
  CudaRuntimeCheck: TNewCheckBox;
  OkButton: TNewButton;
  CancelButton: TNewButton;
begin
  Result := False;
  Form := CreateCustomForm(ScaleX(390), ScaleY(228), False, True);
  try
    Form.Caption := ExpandConstant('{cm:UninstallTitle}');
    Form.BorderStyle := bsDialog;

    HeaderLabel := TNewStaticText.Create(Form);
    HeaderLabel.Parent := Form;
    HeaderLabel.Left := ScaleX(16);
    HeaderLabel.Top := ScaleY(14);
    HeaderLabel.Width := Form.ClientWidth - ScaleX(32);
    HeaderLabel.Height := ScaleY(36);
    HeaderLabel.WordWrap := True;
    HeaderLabel.Caption := ExpandConstant('{cm:UninstallPrompt}');

    ConfigCheck := TNewCheckBox.Create(Form);
    ConfigCheck.Parent := Form;
    ConfigCheck.Left := ScaleX(16);
    ConfigCheck.Top := ScaleY(62);
    ConfigCheck.Width := Form.ClientWidth - ScaleX(32);
    ConfigCheck.Height := ScaleY(20);
    ConfigCheck.Caption := ExpandConstant('{cm:UninstallConfig}');
    ConfigCheck.Checked := True;

    LogsCheck := TNewCheckBox.Create(Form);
    LogsCheck.Parent := Form;
    LogsCheck.Left := ScaleX(16);
    LogsCheck.Top := ScaleY(90);
    LogsCheck.Width := Form.ClientWidth - ScaleX(32);
    LogsCheck.Height := ScaleY(20);
    LogsCheck.Caption := ExpandConstant('{cm:UninstallLogs}');
    LogsCheck.Checked := True;

    ModelsCheck := TNewCheckBox.Create(Form);
    ModelsCheck.Parent := Form;
    ModelsCheck.Left := ScaleX(16);
    ModelsCheck.Top := ScaleY(118);
    ModelsCheck.Width := Form.ClientWidth - ScaleX(32);
    ModelsCheck.Height := ScaleY(20);
    ModelsCheck.Caption := ExpandConstant('{cm:UninstallModels}');
    ModelsCheck.Checked := True;

    CudaRuntimeCheck := TNewCheckBox.Create(Form);
    CudaRuntimeCheck.Parent := Form;
    CudaRuntimeCheck.Left := ScaleX(16);
    CudaRuntimeCheck.Top := ScaleY(146);
    CudaRuntimeCheck.Width := Form.ClientWidth - ScaleX(32);
    CudaRuntimeCheck.Height := ScaleY(20);
    CudaRuntimeCheck.Caption := ExpandConstant('{cm:UninstallCudaRuntime}');
    CudaRuntimeCheck.Checked := False;

    OkButton := TNewButton.Create(Form);
    OkButton.Parent := Form;
    OkButton.Left := Form.ClientWidth - ScaleX(190);
    OkButton.Top := Form.ClientHeight - ScaleY(38);
    OkButton.Width := ScaleX(80);
    OkButton.Caption := SetupMessage(msgButtonOK);
    OkButton.ModalResult := mrOk;
    OkButton.Default := True;

    CancelButton := TNewButton.Create(Form);
    CancelButton.Parent := Form;
    CancelButton.Left := Form.ClientWidth - ScaleX(100);
    CancelButton.Top := Form.ClientHeight - ScaleY(38);
    CancelButton.Width := ScaleX(80);
    CancelButton.Caption := ExpandConstant('{cm:BtnCancel}');
    CancelButton.ModalResult := mrCancel;
    CancelButton.Cancel := True;

    if Form.ShowModal() = mrOk then
    begin
      RemoveConfig := ConfigCheck.Checked;
      RemoveLogs := LogsCheck.Checked;
      RemoveModels := ModelsCheck.Checked;
      RemoveCudaRuntime := CudaRuntimeCheck.Checked;
      Result := True;
    end;
  finally
    Form.Free();
  end;
end;

procedure DeleteDirIfPresent(const DirPath: string);
begin
  if DirExists(DirPath) then
    DelTree(DirPath, True, True, True);
end;

procedure TryRemoveDirIfEmpty(const DirPath: string);
begin
  if DirExists(DirPath) then
    RemoveDir(DirPath);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDataDir: string;
  LocalDataDir: string;
begin
  if CurUninstallStep = usUninstall then
  begin
    if not EnsureWhisplyClosed() then
      Abort();

    RegDeleteValue(HKEY_CURRENT_USER, 'Software\Microsoft\Windows\CurrentVersion\Run', 'Whisply');

    if UninstallSilent then
    begin
      RemoveConfig := True;
      RemoveLogs := True;
      RemoveModels := True;
      RemoveCudaRuntime := False;
    end
    else
    begin
      if not ShowUninstallDataDialog() then
        Abort();
    end;

    AppDataDir := ExpandConstant('{userappdata}\Whisply');
    LocalDataDir := ExpandConstant('{localappdata}\Whisply');

    if RemoveConfig then
      DeleteDirIfPresent(AppDataDir);
    if RemoveLogs then
      DeleteDirIfPresent(LocalDataDir + '\logs');
    if RemoveModels then
      DeleteDirIfPresent(LocalDataDir + '\models');
    if RemoveCudaRuntime then
      DeleteDirIfPresent(LocalDataDir + '\cuda_runtime');

    TryRemoveDirIfEmpty(LocalDataDir);
  end;
end;
