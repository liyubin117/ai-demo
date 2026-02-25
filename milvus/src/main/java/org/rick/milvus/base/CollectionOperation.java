package org.rick.milvus.base;

import io.milvus.param.Constant;
import io.milvus.v2.common.ConsistencyLevel;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.*;
import io.milvus.v2.service.collection.response.DescribeCollectionResp;
import io.milvus.v2.service.collection.response.ListCollectionsResp;
import io.milvus.v2.service.partition.request.HasPartitionReq;
import io.milvus.v2.service.partition.request.ListPartitionsReq;
import io.milvus.v2.service.partition.request.LoadPartitionsReq;
import io.milvus.v2.service.partition.request.ReleasePartitionsReq;
import org.apache.commons.lang3.StringUtils;

import java.util.*;

import static org.rick.milvus.base.Common.initClient;

public class CollectionOperation {
    public static void main(String[] args) {
        // 创建集合并加载
        createCollection("rick_db", "rick_collection_dynamic_field");
        loadCollection("rick_db", "rick_collection_dynamic_field");
        System.out.println("[加载状态] rick_collection_dynamic_field: " + getCollectionLoadState("rick_db", "rick_collection_dynamic_field", null));
        releaseLoad("rick_db", "rick_collection_dynamic_field");
        System.out.println("[释放后加载状态] rick_collection_dynamic_field: " + getCollectionLoadState("rick_db", "rick_collection_dynamic_field", null));

        // 默认数据库操作
        createCollection("default", "collection1");
        System.out.println("[集合名称列表]: ");
        getCollectionNames().forEach(System.out::println);
        System.out.println("[分区列表] collection1: ");
        getCollectionPartitions("default", "collection1").forEach(System.out::println);
        System.out.println("[是否存在默认分区] collection1._default: " + isCollectionPartitionExisted("default", "collection1", "_default"));
        System.out.println("[加载状态] collection1._default: " + getCollectionLoadState("default", "collection1", "_default"));
        loadCollectionPartition("default", "collection1", "_default");
        System.out.println("[加载后状态] collection1._default: " + getCollectionLoadState("default", "collection1", "_default"));
        releasePartitionLoad("default", "collection1", "_default");
        System.out.println("[释放后状态] collection1._default: " + getCollectionLoadState("default", "collection1", "_default"));

        // 集合描述和修改
        System.out.println("[集合描述] collection1: " + descCollection());
        alterCollection("default", "collection1");
        System.out.println("[修改后集合描述] collection1: " + descCollection());

        // 删除和重命名集合
        dropCollection("default", "collection_new");
        dropCollection("rick_db", "rick_collection_dynamic_field");
        renameCollection("collection1", "collection_new");
        System.out.println("[重命名后集合列表]: ");
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
                .consistencyLevel(ConsistencyLevel.EVENTUALLY) // 为集合中的搜索和查询设置一致性级别，按一致性严格程度从高到低依次是STRONG SESSION BOUNDED EVENTUALLY，严格程度取决于客户端设定的保证时间与服务端的服务时间的gap
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

    public static Boolean getCollectionLoadState(String db, String collection, String part) {
        GetLoadStateReq.GetLoadStateReqBuilder reqBuilder = GetLoadStateReq.builder()
                .databaseName(db)
                .collectionName(collection)
                ;
        if (StringUtils.isNotEmpty(part)) {
            reqBuilder.partitionName(part);
        }
        return initClient().getLoadState(reqBuilder.build());
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

    /**
     * 分区是一个 Collection 的子集。每个分区与其父集合共享相同的数据结构，但只包含集合中的一个数据子集
     * 创建一个 Collection 时，Milvus 也会在该 Collection 中创建一个名为_default 的分区。如果不添加其他分区，所有插入到 Collections 中的实体都会进入默认分区，所有搜索和查询也都在默认分区内进行。
     * 可以添加更多分区，并根据特定条件将实体插入其中。这样就可以限制在某些分区内进行搜索和查询，从而提高搜索性能。
     * 一个 Collections 最多可以有 1,024 个分区。
     */
    public static List<String> getCollectionPartitions(String db, String collection) {
        ListPartitionsReq req = ListPartitionsReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .build();
        return initClient().listPartitions(req);
    }

    public static boolean isCollectionPartitionExisted(String db, String collection, String part) {
        HasPartitionReq req = HasPartitionReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .partitionName(part)
                .build();
        return initClient().hasPartition(req);
    }

    public static void releasePartitionLoad(String db, String collection, String part) {
        ReleasePartitionsReq releasePartitionsReq = ReleasePartitionsReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .partitionNames(Collections.singletonList(part))
                .build();
        initClient().releasePartitions(releasePartitionsReq);
    }

    public static void loadCollectionPartition(String db, String collection, String part) {
        LoadPartitionsReq loadPartitionsReq = LoadPartitionsReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .partitionNames(Collections.singletonList(part))
                .build();
        initClient().loadPartitions(loadPartitionsReq);
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
