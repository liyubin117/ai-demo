package org.rick.milvus.base;

import io.milvus.param.Constant;
import io.milvus.v2.common.ConsistencyLevel;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.*;
import io.milvus.v2.service.collection.response.DescribeCollectionResp;
import io.milvus.v2.service.collection.response.ListCollectionsResp;

import java.util.*;

import static org.rick.milvus.base.Common.initClient;

public class CollectionOperation {
    public static void main(String[] args) {
        createCollection("rick_db", "rick_collection_dynamic_field");
        loadCollection("rick_db", "rick_collection_dynamic_field");
        System.out.println(getCollectionLoadState("rick_db", "rick_collection_dynamic_field"));
        releaseLoad("rick_db", "rick_collection_dynamic_field");
        System.out.println(getCollectionLoadState("rick_db", "rick_collection_dynamic_field"));

        createCollection("default", "collection1");
        getCollectionNames().forEach(System.out::println);

        System.out.println(descCollection());
        alterCollection("default", "collection1");
        System.out.println(descCollection());

        dropCollection("default", "collection_new");
        dropCollection("rick_db", "rick_collection_dynamic_field");
        renameCollection("collection1", "collection_new");
        getCollectionNames().forEach(System.out::println);
    }

    public static CreateCollectionReq.CollectionSchema createSchema() {
        // 3.1 Create schema
        CreateCollectionReq.CollectionSchema schema = CreateCollectionReq.CollectionSchema.builder()
                /**
                 * 启用动态字段
                 * 允许你通过动态字段这一特殊功能，插入结构灵活、不断变化的实体。
                 * 该字段以名为$meta 的隐藏 JSON 类型字段实现，它会自动以键值对的json形式存储数据中任何未在 Collections Schema 中明确定义的字段
                 */
                .enableDynamicField(true)
                .build();

        // 3.2 Add fields to schema
        schema.addField(AddFieldReq.builder()
                .fieldName("my_id")
                .dataType(DataType.Int64)
                .isPrimaryKey(true)
                .autoID(true)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName("my_vector")
                .dataType(DataType.FloatVector)
                .dimension(5)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName("my_varchar")
                .dataType(DataType.VarChar)
                .maxLength(512)
//                .isPartitionKey(true)
                .build());

        return schema;
    }

    // 索引
    public static List<IndexParam> indexParams() {
        // 3.3 Prepare index parameters
        IndexParam indexParamForIdField = IndexParam.builder()
                .fieldName("my_id")
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .build();

        IndexParam indexParamForVectorField = IndexParam.builder()
                .fieldName("my_vector")
                .indexType(IndexParam.IndexType.AUTOINDEX)
                .metricType(IndexParam.MetricType.COSINE)
                .build();

        List<IndexParam> indexParams = new ArrayList<>();
        indexParams.add(indexParamForIdField);
        indexParams.add(indexParamForVectorField);

        return indexParams;
    }

    public static void createCollection(String db, String collection) {
        // 3.4 Create a collection with schema and index parameters
        CreateCollectionReq customizedSetupReq1 = CreateCollectionReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .collectionSchema(createSchema())
                .indexParams(indexParams()) // 若索引，则在创建时会自动加载
                .numShards(3) // 设置分片数
                .property(Constant.MMAP_ENABLED, "true") // 启用mmap，默认true。允许 Milvus 将原始字段数据映射到内存中，而不是完全加载它们。这样可以减少内存占用，提高 Collections 的容量
                .property(Constant.TTL_SECONDS, "86400") // 一旦 TTL 超时，Milvus 就会删除 Collection 中的实体。删除是异步的，这表明在删除完成之前，搜索和查询仍然可以进行
                .consistencyLevel(ConsistencyLevel.EVENTUALLY) // 为集合中的搜索和查询设置一致性级别
                .build();

        initClient().createCollection(customizedSetupReq1);
    }

    /**
     * 加载集合是在集合中进行相似性搜索和查询的前提
     * 加载 Collections 时，Milvus 会将索引文件和所有字段的原始数据加载到内存中，以便快速响应搜索和查询。在载入 Collections 后插入的实体会自动编入索引并载入
     */
    public static void loadCollection(String db, String collection) {
        LoadCollectionReq req = LoadCollectionReq.builder()
                .databaseName(db)
                .collectionName(collection)
                // 加载特定字段在内测中，生产慎用，如果选择加载特定字段，只有load_fields 中包含的字段才能用作搜索和查询中的过滤器和输出字段。您应始终在load_fields 中包含主字段和至少一个向量字段的名称
                .loadFields(Arrays.asList("my_id", "my_vector"))
                .build();
        initClient().loadCollection(req);
    }

    public static Boolean getCollectionLoadState(String db, String collection) {
        GetLoadStateReq customSetupLoadStateReq1 = GetLoadStateReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .build();

        return initClient().getLoadState(customSetupLoadStateReq1);
    }

    public static void releaseLoad(String db, String collection) {
        ReleaseCollectionReq releaseCollectionReq = ReleaseCollectionReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .build();

        initClient().releaseCollection(releaseCollectionReq);
    }

    /**
     * 只能看到default库的
     */
    public static List<String> getCollectionNames() {
        ListCollectionsResp resp = initClient().listCollections();
        return resp.getCollectionNames();
    }

    public static DescribeCollectionResp descCollection() {
        DescribeCollectionReq request = DescribeCollectionReq.builder()
                .collectionName("collection1")
                .build();
        return initClient().describeCollection(request);
    }

    public static void renameCollection(String srcName, String destName) {
        RenameCollectionReq renameCollectionReq = RenameCollectionReq.builder()
                .collectionName(srcName)
                .newCollectionName(destName)
                .build();
        initClient().renameCollection(renameCollectionReq);
    }

    public static void alterCollection(String db, String collection) {
        Map<String, String> properties = new HashMap<>();
        properties.put("collection.ttl.seconds", "60");
//        properties.put("mmap.enabled", "True");
        properties.put("dynamicfield.enabled", "True");
        properties.put("timezone", "Asia/Shanghai");
        /**
         * 需要先在schema里有partition key。
         * 启用分区密钥隔离后，Milvus 会根据分区密钥值对实体进行分组，并为每个分组创建单独的索引。
         * 收到搜索请求后，Milvus 会根据过滤条件中指定的 Partition Key 值定位索引，并将搜索范围限制在索引所包含的实体内，从而避免在搜索过程中扫描不相关的实体，大大提高搜索性能。
         */
//        properties.put("partitionkey.isolation", "True");
        /**
         * 当为 Collections 启用自动 ID 时，是否允许 Collections 接受用户提供的主键值。
         *  设置为"true"时：插入、向上插入和批量导入时，如果存在用户提供的主键，则使用用户提供的主键；否则，将自动生成主键值。
         *  设置为"false"时：用户提供的主键值将被拒绝或忽略，主键值始终是自动生成的。默认值为"false"。
         */
        properties.put("allow_insert_auto_id", "True");

        // 不能同时在Map里加多个键值属性，milvus不支持。。
        for (Map.Entry<String, String> entry: properties.entrySet()) {
            alterCollection(db, collection, entry.getKey(), entry.getValue());
        }

    }
    public static void alterCollection(String db, String collection, String key, String value) {
        Map<String, String> properties = Collections.singletonMap(key, value);

        AlterCollectionReq alterCollectionReq = AlterCollectionReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .properties(properties)
                .build();

        initClient().alterCollection(alterCollectionReq);
    }

    public static void dropCollection(String db, String collection) {
        DropCollectionReq dropQuickSetupParam = DropCollectionReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .build();

        initClient().dropCollection(dropQuickSetupParam);
    }
}
