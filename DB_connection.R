
## Connect to SQL Server using Windows Authentication

# Step 1: Install RODBC package through R Studio Tools, then it may ask you to download the source file from a link
# Step 2 (if Step 1 cannot install RODBC completely): If so
install.packages("[path of the downloaded .tar.gz]/RODBC_1.3-13.tar.gz", repos = NULL, type="source")

# Step 3 (if Step 2 failed): if cannot install RODBC from source, type command line "brew install unixodbc" in your terminal
# then run Step 2 again

library(RODBC)
db_handler <- odbcDriverConnect('driver={SQL Server};server=[server name];database=[database name];trusted_connection=true')
q1 <- sqlQuery(db_handler, "[normal tsql query]")
close(db_handler)



#------------------------------------#

## Connect to Oracle

library(rJava)
library(RJDBC)
drv <- JDBC("oracle.jdbc.OracleDriver", classPath="[Your file path for the downloaded ojdbc .jar file]\\ojdbc6.jar", " ")
con <- dbConnect(drv, "jdbc:oracle:thin:@[host name]:[port number]:[database name]", "[user name]", "[password]")
# NOTE: if the query contains double quotes, use \ before each double quote
# ALSO NOTE: In Oracle we should use ; to end the query, but in R Oracle query, ; will lead to error
d_oracle <- dbGetQuery(con, "[normal Oracle sql query]")  
dbDisconnect(con)
