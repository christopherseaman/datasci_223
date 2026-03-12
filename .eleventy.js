const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
const markdownIt = require("markdown-it");
const markdownItTaskLists = require("markdown-it-task-lists");
const markdownItCallouts = require("markdown-it-obsidian-callouts");

module.exports = function (eleventyConfig) {
  eleventyConfig.addPlugin(syntaxHighlight);

  const md = markdownIt({ html: true, linkify: true, typographer: true })
    .use(markdownItTaskLists)
    .use(markdownItCallouts);
  eleventyConfig.setLibrary("md", md);

  // Passthrough copy — lecture media and CSS
  eleventyConfig.addPassthroughCopy("css");
  // Copy XX/media/ → XX/media/ so relative paths work from /XX/
  for (const id of ["01","02","03","04","05","06","07","08","09","10","11"]) {
    eleventyConfig.addPassthroughCopy({
      [`${id}/media`]: `${id}/media`,
    });
  }

  // Computed data — layout, title, and permalink from file paths
  eleventyConfig.addGlobalData("eleventyComputed", {
    layout: (data) => data.layout || "layout.njk",
    title: (data) => {
      if (data.title) return data.title;
      const inputPath = data.page?.inputPath || "";
      const match = inputPath.match(/(\d{2})\/lecture_\d{2}\.md$/);
      if (!match) return undefined;
      const nav = require("./_data/nav.js");
      const lecture = nav.lectures.find((l) => l.id === match[1]);
      return lecture ? `${match[1]}: ${lecture.label}` : undefined;
    },
    navHidden: (data) => {
      // Frontmatter `nav_visible: true` overrides the default
      if (data.nav_visible === true) return false;
      if (data.nav_visible === false) return true;
      // Lectures hide nav by default; other pages show it
      return /^\.\/?(\d{2})\/lecture_\d{2}\.md$/.test(data.page?.inputPath || "");
    },
    permalink: (data) => {
      if (data.permalink) return data.permalink;
      const inputPath = data.page?.inputPath || "";
      const lectureMatch = inputPath.match(/(\d{2})\/lecture_\d{2}\.md$/);
      if (lectureMatch) return `/${lectureMatch[1]}/`;
      return undefined;
    },
  });

  // Lecture collection for index page listing
  eleventyConfig.addCollection("lectures", (collectionApi) => {
    return collectionApi
      .getFilteredByGlob("*/lecture_*.md")
      .sort((a, b) => {
        const numA = a.inputPath.match(/(\d{2})\/lecture_\d{2}\.md$/)?.[1] || "0";
        const numB = b.inputPath.match(/(\d{2})\/lecture_\d{2}\.md$/)?.[1] || "0";
        return numA.localeCompare(numB);
      });
  });

  eleventyConfig.addWatchTarget("css/");

  return {
    dir: { input: ".", output: "_site", includes: "_includes", data: "_data" },
    markdownTemplateEngine: "njk",
    pathPrefix: process.env.ELEVENTY_PATH_PREFIX || "/",
  };
};
