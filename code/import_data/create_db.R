install.packages("pitchRx")
install.packages("dbplyr")
install.packages("RSQLite")
library(dplyr)
library(dbplyr)
library(pitchRx)
db <- src_sqlite("pitchfx.sqlite3", create = T)
scrape(
    start = "2013-04-01",
    end = "2013-04-10",
    connect = db$con
)
