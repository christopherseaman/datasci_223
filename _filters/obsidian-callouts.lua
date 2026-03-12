local stringify = (require "pandoc.utils").stringify

function BlockQuote(el)
  local start = el.content[1]
  if (start.t == "Para" and start.content[1].t == "Str" and
      start.content[1].text:match("^%[!%w+%][-+]?$")) then
    local _, _, ctype = start.content[1].text:find("%[!(%w+)%]")
    local titlevar = stringify(start.content):match("^%[!%w+%](.-)$")
    el.content:remove(1)
    return quarto.Callout({
      type = ctype:lower(),
      title = titlevar or "",
      content = el.content
    })
  else
    return el
  end
end
