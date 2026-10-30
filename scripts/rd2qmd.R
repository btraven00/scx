# Write the R reference pages of the docs site from r/picklerick/man, as
# `quartodoc build` does for Python: one page per help topic plus an index,
# grouped as the `r-reference` block of site/_quarto.yml lists them. Run by
# scripts/render_docs.sh before `quarto render`.
#
#   Rscript scripts/rd2qmd.R <package dir> <site dir>
#
# Each Rd file goes through tools::Rd2HTML and then pandoc (Quarto's) to
# Markdown, so the sections become real headings and the code is highlighted.
# Mentions of other topics, `read_h5ad()` say, link to their pages.

args <- commandArgs(trailingOnly = TRUE)
pkg <- args[[1]]
site <- args[[2]]

config <- yaml::read_yaml(file.path(site, "_quarto.yml"))[["r-reference"]]
out_dir <- file.path(site, config$dir)
unlink(out_dir, recursive = TRUE)
dir.create(out_dir, recursive = TRUE)

quarto <- Sys.which("quarto")
if (!nzchar(quarto)) stop("quarto is not on PATH")

rd_tag <- function(rd, tag) {
  Filter(function(x) identical(attr(x, "Rd_tag"), tag), rd)
}
rd_text <- function(x) trimws(paste(unlist(x), collapse = ""))

topics <- unlist(lapply(config$sections, `[[`, "contents"))
rds <- setNames(lapply(topics, function(topic) {
  tools::parse_Rd(file.path(pkg, "man", paste0(topic, ".Rd")))
}), topics)

# The names a topic documents, as the package exports them: read_zarr is on
# read_h5ad's page, while S4 method aliases such as `dim,...-method` are left
# out. A topic that documents no export (AnnDataLight) goes by its own name.
exports <- sub("^export\\((.*)\\)$", "\\1",
               grep("^export\\(", readLines(file.path(pkg, "NAMESPACE")), value = TRUE))
topic_names <- lapply(setNames(topics, topics), function(topic) {
  aliases <- vapply(rd_tag(rds[[topic]], "\\alias"), rd_text, "")
  named <- intersect(aliases, exports)
  if (length(named)) named else topic
})

# Every name (read_zarr, say) points at the page of its topic (read_h5ad).
alias_page <- unlist(lapply(topics, function(topic) {
  setNames(rep(topic, length(topic_names[[topic]])), topic_names[[topic]])
}))

# Only the call form, `uns()`: a bare `uns` is as likely to mean the AnnData
# slot as the function.
link_topics <- function(md) {
  for (alias in names(alias_page)) {
    shown <- paste0(alias, "()")
    pattern <- paste0("(?<!\\[)`", gsub("([.()])", "\\\\\\1", shown), "`")
    page <- paste0(alias_page[[alias]], ".qmd")
    md <- gsub(pattern, paste0("[`", shown, "`](", page, ")"), md, perl = TRUE)
  }
  md
}

for (topic in topics) {
  html <- tempfile(fileext = ".html")
  tools::Rd2HTML(rds[[topic]], out = html, standalone = FALSE, package = "picklerick")
  md <- system2(quarto, c("pandoc", "-f", "html", "-t", "gfm", "--wrap=none", html),
                stdout = TRUE)
  md <- md[!grepl("^<span id=\"topic\\+", md)]       # Rd's alias anchors
  md <- md[!grepl("^## (Not run|End\\(Not run\\))", md)]
  md <- gsub("⁠", "", md)                         # word joiners from \verb
  md <- sub("^``` R$", "```r", md)
  title <- sub("^## ", "", md[[1]])
  md <- link_topics(md[-1])
  writeLines(c("---", paste0("title: ", shQuote(title, "cmd")), "---", "", md),
             file.path(out_dir, paste0(topic, ".qmd")))
}

index <- c("---", paste0("title: ", shQuote(config$title, "cmd")), "---", "")
for (section in config$sections) {
  index <- c(index, paste("##", section$title), "",
             "| | |", "|---|---|")
  for (topic in section$contents) {
    links <- paste0("[`", topic_names[[topic]], "`](", topic, ".qmd)", collapse = ", ")
    index <- c(index, paste0("| ", links, " | ", rd_text(rd_tag(rds[[topic]], "\\title")), " |"))
  }
  index <- c(index, "")
}
writeLines(index, file.path(out_dir, "index.qmd"))
