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

  -- Build title from remaining inlines in the first paragraph (after [!TYPE] and any space)
  local title_inlines = pandoc.List()
  local skip = true
  for i, inline in ipairs(start.content) do
    if skip then
      if i == 1 then
        -- skip the [!TYPE] token
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

  local title = pandoc.utils.stringify(pandoc.Inlines(title_inlines))

  -- Body is everything after the first paragraph
  local body = pandoc.List()
  for i = 2, #el.content do
    body:insert(el.content[i])
  end

  return quarto.Callout({
    type = ctype:lower(),
    title = title,
    content = body
  })
end
