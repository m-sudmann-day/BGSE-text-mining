
library(ggplot2)

orig <- read.csv("C:\\OneDrive\\Career\\github\\BGSE-text-mining\\week2_hw\\output - Copy.csv")

one_plot <- function(analysis, title, pair1, pair2, pair3)
{
  dfAA <- orig[orig$analysis_num==analysis,c(1,2,3)]
  colnames(dfAA)[3] <- "value"
  dfAA$pair <- pair1
  
  dfAB <- orig[orig$analysis_num==analysis,c(1,2,4)]
  colnames(dfAB)[3] <- "value"
  dfAB$pair = pair2
  
  dfBB <- orig[orig$analysis_num==analysis,c(1,2,5)]
  colnames(dfBB)[3] <- "value"
  dfBB$pair = pair3
  
  df <- rbind(dfAA, dfAB, dfBB)
  
  p <- ggplot(data=df)
  p <- p + ggtitle(title)
  p <- p + xlab("Singular Values") + ylab("Cosine Similarity")
  p <- p + theme(legend.text = element_text("Legend"))
  p <- p + geom_line(aes(x=max_singular_values, y=value, color=pair))
  ggsave(paste("analysis", analysis, ".png", sep=""))
  p
}

one_plot(1, "Two Most Recent Presidents", "Obama<>Obama", "Obama<>Bush(W)", "Bush(W)<>Bush(W)")

one_plot(2, "Presidents Since 1900, by party", "Dem<>Dem", "Dem<>Rep", "Rep<>Rep")

one_plot(3, "Two Centuries of Presidents", "19th<>19th", "19th<>20th", "20th<>20th")

