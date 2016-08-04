library(RODBC)
db1 <- odbcDriverConnect('driver={SQL Server};server=[SERVER NAME];database=[DATABASE NAME];trusted_connection=true')
db2 <- odbcDriverConnect('driver={SQL Server};server=[SERVER NAME];database=[DATABASE NAME];trusted_connection=true')

d1 <- sqlQuery(db1, "
     select col1
     ,col2
     ,col3
     from db1.table1
")


d2 <- sqlQuery(db2, "
     select col1
     ,col4
     ,col5
     from db2.table2
")


# NOTE!: In R merge() method, both query results (d1, d2 here) must have the same column name as the column will be used to join

result <- merge(x = d1, y = d2, by = "col1", all = F)    # Nature Join (Inner Join)
result <- merge(x = d1, y = d2, by = "col1", all.x = T)  # Left Join
result <- merge(x = d1, y = d2, by = "col1", all.y = T)  # Right Join
result <- merge(x = d1, y = d2, by = "col1", all = T)    # Full Outer Join
