package org.rick.milvus.base;

import io.milvus.param.Constant;
import io.milvus.v2.common.ConsistencyLevel;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.collection.request.GetLoadStateReq;

import java.util.ArrayList;
import java.util.List;

import static org.rick.milvus.base.Common.initClient;

public class SchemaOperation {
    public static void main(String[] args) {
        createCollection("rick_db", "rick_collection_dynamic_field");
    }

    public static CreateCollectionReq.CollectionSchema createSchema() {
        // 3.1 Create schema
        CreateCollectionReq.CollectionSchema schema = CreateCollectionReq.CollectionSchema.builder()
                .enableDynamicField(true) // 启用动态字段
                .build();

        // 3.2 Add fields to schema
        schema.addField(AddFieldReq.builder()
                .fieldName("my_id")
                .dataType(DataType.Int64)
                .isPrimaryKey(true)
                .autoID(false)
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
                .build());

        return schema;
    }

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

        // 3.5 Get load state of the collection
        GetLoadStateReq customSetupLoadStateReq1 = GetLoadStateReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .build();

        Boolean loaded = initClient().getLoadState(customSetupLoadStateReq1);
        System.out.println(loaded);
    }
}
