package org.rick.milvus.base;

import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;

public class Common {
    public static MilvusClientV2 initClient() {
        ConnectConfig config = ConnectConfig.builder()
                .uri("http://localhost:19530")
                .token("root:Milvus")
                .build();
        return new MilvusClientV2(config);
    }
}
