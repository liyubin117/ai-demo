package org.rick.milvus.base;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import io.milvus.param.Constant;
import io.milvus.v2.common.ConsistencyLevel;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.UpsertReq;
import io.milvus.v2.service.vector.request.data.BaseVector;
import io.milvus.v2.service.vector.request.data.BinaryVec;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.request.data.SparseFloatVec;
import io.milvus.v2.service.vector.response.InsertResp;
import io.milvus.v2.service.vector.response.SearchResp;
import io.milvus.v2.service.vector.response.UpsertResp;

import java.util.*;

import static org.rick.milvus.base.CollectionDdl.*;
import static org.rick.milvus.base.Common.initClient;

public class CollectionDml {
    public static void main(String[] args) {
        if (getCollectionNames().contains("dml")) {
            dropCollection("default", "dml");
        }

        createCollection("default", "dml");
        insert("default", "dml");
        // upsert需要手动指定主键，且需要schema的autoID为false
        upsert("default", "dml");

        System.out.println("[密集向量搜索结果]: " + search("default", "dml", "embedding", new FloatVec(new float[]{0.1f, 0.3f, 0.3f, 0.4f})));

        boolean[] boolArray = {true, false, false, true, true, false, true, true, false, true, false, false, true, true, false, true};
        BinaryVec queryVector = new BinaryVec(convertBoolArrayToBytes(boolArray));
        System.out.println("[二进制向量搜索结果]: " + search("default", "dml", "binary_vector", queryVector));

        SortedMap<Long, Float> sparse = new TreeMap<>();
        sparse.put(1L, 0.2f);
        sparse.put(50L, 0.4f);
        sparse.put(1000L, 0.7f);
        SparseFloatVec sparseFloatVec = new SparseFloatVec(sparse);
        System.out.println("[稀疏向量搜索结果]: " + search("default", "dml", "sparse_vector", sparseFloatVec));
    }

    public static void createCollection(String db, String collection) {
        // 3.4 Create a collection with schema and index parameters
        CreateCollectionReq customizedSetupReq1 = CreateCollectionReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .collectionSchema(createSchema())
                .numShards(3) // 设置分片数
                .indexParams(indexParams()) // 为了加速语义搜索，建议为向量字段创建索引。索引可以大大提高大规模向量数据的检索效率
                .property(Constant.MMAP_ENABLED, "true") // 启用mmap，默认true。允许 Milvus 将原始字段数据映射到内存中，而不是完全加载它们。这样可以减少内存占用，提高 Collections 的容量
                .property(Constant.TTL_SECONDS, "86400") // 一旦 TTL 超时，Milvus 就会删除 Collection 中的实体。删除是异步的，这表明在删除完成之前，搜索和查询仍然可以进行
                .consistencyLevel(ConsistencyLevel.EVENTUALLY) // 为集合中的搜索和查询设置一致性级别，按一致性严格程度从高到低依次是STRONG SESSION BOUNDED EVENTUALLY，严格程度取决于客户端设定的保证时间与服务端的服务时间的gap
                .build();

        initClient().createCollection(customizedSetupReq1);
    }

    public static CreateCollectionReq.CollectionSchema createSchema() {
        CreateCollectionReq.CollectionSchema collectionSchema = CreateCollectionReq.CollectionSchema.builder()
                .build();
        collectionSchema.addField(AddFieldReq.builder()
                .fieldName("product_id")
                .dataType(DataType.VarChar)
                .isPrimaryKey(true)
                .autoID(true)
                .maxLength(100)
                .build());
        collectionSchema.addField(AddFieldReq.builder()
                .fieldName("embedding") // 密集向量
                .dataType(DataType.FloatVector)
                .dimension(4)
                .build());
        collectionSchema.addField(AddFieldReq.builder()
                .fieldName("category")
                .dataType(DataType.VarChar)
                .maxLength(1000)
                .build());
        collectionSchema.addField(AddFieldReq.builder()
                .fieldName("binary_vector") // 二进制向量
                .dataType(DataType.BinaryVector)
                .dimension(16)
                .build());
        collectionSchema.addField(AddFieldReq.builder()
                .fieldName("sparse_vector") // 稀疏向量
                .dataType(DataType.SparseFloatVector)
                .build());
        collectionSchema.addField(AddFieldReq.builder()
                .fieldName("text")
                .dataType(DataType.VarChar)
                .maxLength(65535)
                .enableAnalyzer(true)
                .build());
        return collectionSchema;
    }

    public static List<IndexParam> indexParams() {

        IndexParam indexParamForVectorField = IndexParam.builder()
                .fieldName("embedding")
                .indexType(IndexParam.IndexType.AUTOINDEX) // 旨在平滑向量搜索的学习曲线
                .metricType(IndexParam.MetricType.IP) // 使用内积作为距离度量
                .build();
        IndexParam indexParamForBinaryVectorField = IndexParam.builder()
                .fieldName("binary_vector")
                .indexType(IndexParam.IndexType.AUTOINDEX) // 旨在平滑向量搜索的学习曲线
                .metricType(IndexParam.MetricType.HAMMING) // 使用汉明距离作为距离度量
                .build();
        Map<String,Object> extraParams = new HashMap<>();
        extraParams.put("inverted_index_algo", "DAAT_MAXSCORE"); // Algorithm used for building and querying the index
        IndexParam indexParamForSparseVectorField = IndexParam.builder()
                .fieldName("sparse_vector")
                .indexName("sparse_inverted_index")
                .indexType(IndexParam.IndexType.SPARSE_INVERTED_INDEX)
                .metricType(IndexParam.MetricType.IP)
                .extraParams(extraParams)
                .build();

        List<IndexParam> indexParams = new ArrayList<>();
        indexParams.add(indexParamForVectorField);
        indexParams.add(indexParamForBinaryVectorField);
        indexParams.add(indexParamForSparseVectorField);

        return indexParams;
    }

    public static void insert(String db, String collection) {
        List<JsonObject> rows = new ArrayList<>();
        Gson gson = new Gson();
        JsonObject row1 = new JsonObject();
        row1.add("embedding", gson.toJsonTree(new float[]{0.1f, 0.2f, 0.3f, 0.4f}));
        row1.addProperty("category", "book");
        boolean[] boolArray = new boolean[]{true, false, false, true, true, false, true, true, false, true, false, false, true, true, false, true};
        row1.add("binary_vector", gson.toJsonTree(convertBoolArrayToBytes(boolArray)));
        row1.addProperty("text", "information retrieval is a field of study.");
        SortedMap<Long, Float> sparse = new TreeMap<>();
        sparse.put(1L, 0.5f);
        sparse.put(100L, 0.3f);
        sparse.put(500L, 0.8f);
        row1.add("sparse_vector", gson.toJsonTree(sparse));
        rows.add(row1);

        JsonObject row2 = new JsonObject();
        row2.add("embedding", gson.toJsonTree(new float[]{0.2f, 0.3f, 0.4f, 0.5f}));
        row2.addProperty("category", "toy");
        boolArray = new boolean[]{false, true, false, true, false, true, false, false, true, true, false, false, true, true, false, true};
        row2.add("binary_vector", gson.toJsonTree(convertBoolArrayToBytes(boolArray)));
        row2.addProperty("text", "information retrieval focuses on finding relevant information in large datasets.");
        sparse = new TreeMap<>();
        sparse.put(10L, 0.1f);
        sparse.put(200L, 0.7f);
        sparse.put(1000L, 0.9f);
        row2.add("sparse_vector", gson.toJsonTree(sparse));
        rows.add(row2);

        InsertResp insertResp = initClient().insert(InsertReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .data(rows)
                .build());
        System.out.printf("After insert, Generated IDs: %s\n", insertResp.getPrimaryKeys());
    }

    public static void upsert(String db, String collection) {
        List<JsonObject> rows = new ArrayList<>();
        Gson gson = new Gson();
        JsonObject row1 = new JsonObject();
        row1.addProperty("product_id", "PROD-001");
        row1.add("embedding", gson.toJsonTree(new float[]{0.1f, 0.2f, 0.3f, 0.4f}));
        row1.addProperty("category", "book");
        boolean[] boolArray = new boolean[]{true, true, false, true, true, false, true, false, false, true, false, false, true, true, false, true};
        row1.add("binary_vector", gson.toJsonTree(convertBoolArrayToBytes(boolArray)));
        row1.addProperty("text", "information retrieval is a field of study.");
        SortedMap<Long, Float> sparse = new TreeMap<>();
        sparse.put(10L, 0.5f);
        sparse.put(190L, 0.3f);
        sparse.put(900L, 0.8f);
        row1.add("sparse_vector", gson.toJsonTree(sparse));
        rows.add(row1);

        JsonObject row2 = new JsonObject();
        row2.addProperty("product_id", "PROD-002");
        row2.add("embedding", gson.toJsonTree(new float[]{0.2f, 0.3f, 0.4f, 0.5f}));
        row2.addProperty("category", "toy");
        boolArray = new boolean[]{true, true, false, true, false, true, false, true, true, true, false, false, true, true, false, true};
        row2.add("binary_vector", gson.toJsonTree(convertBoolArrayToBytes(boolArray)));
        row2.addProperty("text", "information retrieval focuses on finding relevant information in large datasets.");
        sparse = new TreeMap<>();
        sparse.put(120L, 0.1f);
        sparse.put(800L, 0.7f);
        sparse.put(1900L, 0.9f);
        row2.add("sparse_vector", gson.toJsonTree(sparse));
        rows.add(row2);

        UpsertResp upsertResp = initClient().upsert(UpsertReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .data(rows)
                .build());
        System.out.printf("After upsert, Generated IDs: %s\n", upsertResp.getPrimaryKeys());
    }

    public static SearchResp search(String db, String collection, String annsField, BaseVector queryVector) {
        Map<String, Object> searchParams = new HashMap<>();
        searchParams.put("nprobe", 10); // 用于控制相似最近邻搜索的精度
        searchParams.put("drop_ratio_search", 0.2); // 用于控制在近似最近邻搜索（ANN）过程中跳过部分数据的比例，以提高搜索效率，但可能会略微降低精度

        SearchResp searchR = initClient().search(SearchReq.builder()
                .databaseName(db)
                .collectionName(collection)
                .data(Collections.singletonList(queryVector))
                .annsField(annsField)
                .searchParams(searchParams)
                .topK(2)
                .outputFields(Arrays.asList("product_id", annsField))
                .build());
        return searchR;
    }

    private static byte[] convertBoolArrayToBytes(boolean[] booleanArray) {
        byte[] byteArray = new byte[booleanArray.length / Byte.SIZE];
        for (int i = 0; i < booleanArray.length; i++) {
            if (booleanArray[i]) {
                int index = i / Byte.SIZE;
                int shift = i % Byte.SIZE;
                byteArray[index] |= (byte) (1 << shift);
            }
        }

        return byteArray;
    }
}
