package org.rick.milvus.base;

import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;

public class Common {
    public static MilvusClientV2 initClient() {
        ConnectConfig config = ConnectConfig.builder()
                .uri("http://localhost:19530")
                .token("root:Milvus")
                .build();
        MilvusClientV2 client = new MilvusClientV2(config);

        // 添加 JVM 关闭钩子，确保程序退出时释放资源
        Runtime.getRuntime().addShutdownHook(new Thread(client::close));

        return client;
    }
}
