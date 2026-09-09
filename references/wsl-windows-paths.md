# Windows & WSL2 Cross-Environment Path Mapping Guide

How to handle redirected user folders and drive mounts across Windows 11 and WSL2.

---

## 1. The Core Problem

On Windows, user folders (`Desktop`, `Documents`, `Downloads`, `Pictures`, `Videos`, `Music`) are frequently relocated to non-C drives (e.g. `D:`, `E:`, `F:`) to save system partition space.
Hardcoding `C:\Users\<username>\Desktop` causes silent failures.

---

## 2. Dynamic Discovery Pattern

Always query Windows for the actual redirected paths via PowerShell:

```powershell
# In PowerShell:
[Environment]::GetFolderPath('Desktop')
[Environment]::GetFolderPath('MyDocuments')
(Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders').'{374DE290-123F-4565-9164-39C4925E467B}' # Downloads
```

---

## 3. WSL2 Mount Translation

In WSL2:
- Windows drive `C:\` mounts at `/mnt/c/`
- Windows drive `D:\` mounts at `/mnt/d/`
- Windows drive `E:\` mounts at `/mnt/e/`

```bash
# Bash helper to translate Windows path to WSL2 mount path
win_to_wsl() {
  echo "$1" | sed 's|\\|/|g; s|^\([A-Za-z]\):|/mnt/\L\1|'
}
```
