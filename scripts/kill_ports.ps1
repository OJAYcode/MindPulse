param(
  [int[]]$Ports = @(8000,3000,3001,3002,3003)
)

foreach ($p in $Ports) {
  $conns = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue
  if ($conns) {
    foreach ($c in $conns) {
      if ($c.OwningProcess -and $c.OwningProcess -ne 0) {
        Write-Output "Stopping PID $($c.OwningProcess) on port $p"
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
      }
    }
  } else {
    Write-Output "No listeners on port $p"
  }
}
