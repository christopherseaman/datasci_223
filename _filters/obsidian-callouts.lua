-- Map Obsidian callout types to Bootstrap alert classes
local alert_map = {
  info = "info",
  note = "info",
  tip = "success",
  hint = "success",
  warning = "warning",
  caution = "warning",
  danger = "danger",
  error = "danger",
  important = "primary",
  question = "secondary",
}

function BlockQuote(el)
  local start = el.content[1]
  if start == nil or start.t ~= "Para" then
    return el
  end

  local first = start.content[1]
  if first == nil or first.t ~= "Str" then
    return el
  end

  local ctype = first.text:match("^%[!(%w+)%][-+]?$")
  if ctype == nil then
    return el
  end

  local alert_class = alert_map[ctype:lower()] or "info"

  -- Build title inlines from the rest of the first paragraph
  local title_inlines = pandoc.List()
  local skip = true
  for i, inline in ipairs(start.content) do
    if skip then
      if i == 1 then
        -- skip [!TYPE]
      elseif inline.t == "Space" or inline.t == "SoftBreak" then
        -- skip whitespace after [!TYPE]
      else
        skip = false
        title_inlines:insert(inline)
      end
    else
      title_inlines:insert(inline)
    end
  end

  -- Build alert content
  local blocks = pandoc.List()

  -- Add title as strong text if present
  if #title_inlines > 0 then
    blocks:insert(pandoc.Para({pandoc.Strong(title_inlines)}))
  end

  -- Add body (everything after first paragraph)
  for i = 2, #el.content do
    blocks:insert(el.content[i])
  end

  -- Wrap in Bootstrap alert div
  local div = pandoc.Div(blocks)
  div.attr = pandoc.Attr("", {"alert", "alert-" .. alert_class}, {role = "alert"})
  return div
end
