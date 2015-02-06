#!/usr/bin/env th
-- Read CSV file

-- Split string
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end


local filePath = 'train.csv'

-- Count number of rows and columns in file
local i = 0
for line in io.lines(filePath) do
  if i == 0 then
    COLS = #line:split(',')
  end
  i = i + 1
end

local ROWS = i - 1  -- Minus 1 because of header

-- Read data from CSV to tensor
local csvFile = io.open(filePath, 'r')
local header = csvFile:read()

local data = torch.Tensor(ROWS, COLS)

local i = 0
for line in csvFile:lines('*l') do
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
    data[i][key] = val
  end
end

csvFile:close()

-- Serialize tensor
local outputFilePath = 'train.th7'
torch.save(outputFilePath, data)

-- Deserialize tensor object
local restored_data = torch.load(outputFilePath)

-- Make test
print(data:size())
print(restored_data:size())
