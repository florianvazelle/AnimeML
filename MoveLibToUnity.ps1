$sourceFile = ".\cpp\build\Debug\example.dll"
$destFile = ".\unity\Assets\Plugins\Win\x86_64\example.dll"

if(Test-Path -path $destFile){
	Remove-Item $destFile
}

Copy-Item -Path $sourceFile -Destination $destFile