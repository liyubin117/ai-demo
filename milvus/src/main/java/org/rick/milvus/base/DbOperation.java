package org.rick.milvus.base;

import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.database.request.*;
import io.milvus.v2.service.database.response.DescribeDatabaseResp;
import io.milvus.v2.service.database.response.ListDatabasesResp;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class DbOperation {
    public static void main(String[] args) throws InterruptedException {
        createDb("rick_db");
        System.out.println(showDb("rick_db"));
        alterDbProp("rick_db", "database.max.collections", "10");
        System.out.println(showDb("rick_db"));
        resetDbProp("rick_db", "database.max.collections");
        System.out.println(showDb("rick_db"));

        initClient().useDatabase("rick_db"); // 在不断开与 Milvus 连接的情况下从一个数据库切换到另一个数据库

        dropDb("rick_db");
    }

    public static MilvusClientV2 initClient() {
        ConnectConfig config = ConnectConfig.builder()
                .uri("http://localhost:19530")
                .token("root:Milvus")
                .build();
        return new MilvusClientV2(config);
    }

    public static void createDb(String db) {
        Map<String, String> properties = new HashMap<>();
        properties.put("database.replica.number", "3");

        CreateDatabaseReq createDatabaseReq = CreateDatabaseReq.builder()
                .databaseName(db)
                .properties(properties)
                .build();
        initClient().createDatabase(createDatabaseReq);
    }

    public static DescribeDatabaseResp showDb(String db) {
        MilvusClientV2 client = initClient();
        ListDatabasesResp listDatabasesResp = client.listDatabases();

        DescribeDatabaseResp descDBResp = client.describeDatabase(DescribeDatabaseReq.builder()
                .databaseName(db)
                .build());
        return descDBResp;
    }

    public static void alterDbProp(String db, String key, String value) {
        initClient().alterDatabaseProperties(AlterDatabasePropertiesReq.builder()
                .databaseName(db)
                .property(key, value)
                .build());
    }

    public static void resetDbProp(String db, String key) {
        initClient().dropDatabaseProperties(DropDatabasePropertiesReq.builder()
                .databaseName(db)
                .propertyKeys(Collections.singletonList(key))
                .build());
    }

    public static void dropDb(String db) {
        initClient().dropDatabase(DropDatabaseReq.builder()
                .databaseName(db)
                .build());
    }

}
