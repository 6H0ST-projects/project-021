"""
Transformation designer agent for generating ETL logic.
"""

from typing import Dict, Any, Optional, List, Set
from pydantic import BaseModel, Field
import json

from sidewinder.agents.base import BaseAgent, BaseAgentState
from sidewinder.core.config import (
    Source, Target, FeatureType, SchemaField,
    TargetSchema, AutoFeatureConfig
)


class TransformationStep(BaseModel):
    """Represents a single transformation step in the pipeline."""
    name: str
    description: str
    code: str
    dependencies: List[str] = Field(default_factory=list)
    layer: str  # bronze, silver, or gold
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    validation_rules: List[str] = Field(default_factory=list)


class TransformerState(BaseAgentState):
    """State for the transformer agent."""
    source: Source
    target: Target
    source_schema: Dict[str, Any]
    data_quality_issues: List[str]
    bronze_layer: Optional[List[TransformationStep]] = None
    silver_layer: Optional[List[TransformationStep]] = None
    gold_layer: Optional[List[TransformationStep]] = None
    error_handling: Dict[str, Any] = Field(default_factory=dict)
    feature_dependencies: Dict[str, Set[str]] = Field(default_factory=dict)


class TransformationDesigner(BaseAgent[TransformerState]):
    """Agent responsible for designing the ETL transformation logic."""

    async def run(self, state: TransformerState) -> TransformerState:
        """
        Design the transformation pipeline with three layers:
        1. Bronze - Raw data ingestion
        2. Silver - Data cleaning and standardization
        3. Gold - Business logic transformations
        
        Args:
            state: Current transformer state
            
        Returns:
            Updated transformer state with transformation logic
        """
        try:
            # Generate transformations for each layer
            state.bronze_layer = self._generate_bronze_layer(state)
            state.silver_layer = self._generate_silver_layer(state)
            state.gold_layer = self._generate_gold_layer(state)
            
            # Set up error handling
            state.error_handling = {
                "retry_attempts": 3,
                "error_thresholds": {
                    "bronze": 0.1,  # Allow 10% error rate in bronze
                    "silver": 0.05,  # Allow 5% error rate in silver
                    "gold": 0.01    # Allow 1% error rate in gold
                },
                "notification_config": {
                    "email": True,
                    "slack": True,
                    "error_tracking": True
                }
            }
            
            state.completed = True
            return state
            
        except Exception as e:
            state.error = str(e)
            return state

    def _generate_bronze_layer(self, state: TransformerState) -> List[TransformationStep]:
        """Generate bronze layer transformations for raw data ingestion."""
        steps = []
        
        # Step 1: Raw Data Reader
        reader_code = self._generate_reader_code(state.source)
        steps.append(
            TransformationStep(
                name="raw_data_reader",
                description="Read raw data from source",
                code=reader_code,
                layer="bronze",
                input_schema=state.source_schema,
                output_schema=state.source_schema,
                validation_rules=[
                    "Verify source accessibility",
                    "Check data format consistency",
                    "Validate raw schema matches expected schema"
                ]
            )
        )
        
        # Step 2: Raw Data Writer with metadata
        writer_code = f"""
def write_to_bronze(spark, df, target):
    # Configure write options
    write_options = {{
        'mode': 'overwrite',
        'partitionBy': target.options.get('partition_columns', []),
        **target.options or {{}}
    }}
    
    try:
        # Add metadata columns
        df = df.withColumn('ingestion_timestamp', current_timestamp()) \\
            .withColumn('source_system', lit('{state.source.type}')) \\
            .withColumn('batch_id', monotonically_increasing_id())
        
        # Write with error handling
        df.write.format(target.format or 'delta') \\
            .options(**write_options) \\
            .save(f"{{target.location}}/bronze")
            
        # Record metadata
        record_metadata(
            layer="bronze",
            location=target.location,
            row_count=df.count(),
            schema=df.schema.json(),
            timestamp=current_timestamp()
        )
    except Exception as e:
        log_error(f"Error writing to bronze layer: {{str(e)}}")
        raise
"""
        steps.append(
            TransformationStep(
                name="bronze_writer",
                description="Write raw data to bronze layer with metadata",
                code=writer_code,
                dependencies=["raw_data_reader"],
                layer="bronze",
                input_schema=state.source_schema,
                output_schema={
                    **state.source_schema,
                    "ingestion_timestamp": "timestamp",
                    "source_system": "string",
                    "batch_id": "long"
                },
                validation_rules=[
                    "Verify write permissions",
                    "Ensure data integrity during write",
                    "Validate row count matches source",
                    "Check metadata columns"
                ]
            )
        )
        
        return steps
    
    def _generate_silver_layer(self, state: TransformerState) -> List[TransformationStep]:
        """Generate silver layer transformations for data cleaning and standardization."""
        steps = []
        
        # Step 1: Data Type Standardization
        type_standardization = self._generate_type_standardization(state.source_schema, state.data_quality_issues)
        steps.append(
            TransformationStep(
                name="type_standardization",
                description="Standardize data types and formats",
                code=type_standardization,
                dependencies=["bronze_writer"],
                layer="silver",
                input_schema=state.source_schema,
                output_schema=self._get_standardized_schema(state.source_schema),
                validation_rules=[
                    "Verify all data type conversions",
                    "Check for data truncation",
                    "Validate format standardization"
                ]
            )
        )
        
        # Step 2: Data Cleaning and Quality
        cleaning_code = f"""
def clean_data(spark, df):
    # Handle missing values based on data quality issues
    quality_issues = {state.data_quality_issues}
    
    for column, dtype in df.dtypes:
        if dtype == 'string':
            # String cleaning
            df = df.withColumn(column, 
                when(col(column).isNull(), '') \\
                .otherwise(trim(regexp_replace(col(column), '[\\\\t\\\\n\\\\r]', ' ')))
            )
            
            # Handle inconsistent casing if detected
            if any('case' in issue.lower() for issue in quality_issues):
                df = df.withColumn(column, lower(col(column)))
                
        elif dtype in ['int', 'double']:
            # Numeric cleaning
            df = df.withColumn(column,
                when(col(column).isNull(), 0) \\
                .when(isnan(col(column)), None) \\
                .otherwise(col(column))
            )
            
            # Handle outliers if detected
            if any('outlier' in issue.lower() for issue in quality_issues):
                stats = df.select(
                    mean(column).alias('mean'),
                    stddev(column).alias('stddev')
                ).collect()[0]
                
                df = df.withColumn(column,
                    when(
                        abs((col(column) - stats.mean) / stats.stddev) > 3,
                        None
                    ).otherwise(col(column))
                )
        
        elif dtype == 'timestamp':
            # Timestamp validation
            df = df.withColumn(column,
                when(
                    col(column) > current_timestamp() + expr('INTERVAL 1 DAY'),
                    None
                ).otherwise(col(column))
            )
    
    # Remove duplicates if detected
    if any('duplicate' in issue.lower() for issue in quality_issues):
        df = df.dropDuplicates()
    
    # Add quality metrics
    df = df.withColumn('quality_score',
        expr('1.0 - (nullCount / size)').over(Window.partitionBy())
    )
    
    return df
"""
        steps.append(
            TransformationStep(
                name="data_cleaning",
                description="Clean and standardize data",
                code=cleaning_code,
                dependencies=["type_standardization"],
                layer="silver",
                input_schema=self._get_standardized_schema(state.source_schema),
                output_schema=self._get_cleaned_schema(state.source_schema),
                validation_rules=[
                    "Verify null handling",
                    "Check outlier treatment",
                    "Validate string standardization",
                    "Verify duplicate removal",
                    "Check quality score calculation"
                ]
            )
        )
        
        # Step 3: Data Validation and Quality Metrics
        validation_code = f"""
def validate_data(spark, df):
    validation_results = []
    quality_metrics = {{}}
    
    # Schema validation
    expected_schema = {self._get_cleaned_schema(state.source_schema)}
    for field_name, field_type in expected_schema.items():
        if field_name not in df.columns:
            validation_results.append(f"Missing field: {{field_name}}")
        elif str(df.schema[field_name].dataType) != field_type:
            validation_results.append(
                f"Type mismatch for {{field_name}}: expected {{field_type}}, got {{df.schema[field_name].dataType}}"
            )
    
    # Data quality checks
    for column in df.columns:
        stats = df.select(
            count(when(col(column).isNull(), 1)).alias('null_count'),
            count('*').alias('total_count'),
            countDistinct(column).alias('distinct_count')
        ).collect()[0]
        
        null_rate = stats.null_count / stats.total_count
        distinct_rate = stats.distinct_count / stats.total_count
        
        quality_metrics[column] = {{
            'null_rate': null_rate,
            'distinct_rate': distinct_rate,
            'quality_score': 1.0 - null_rate
        }}
        
        # Check against thresholds
        if null_rate > {state.target.quality_thresholds.get('null_rate', 0.1)}:
            validation_results.append(f"High null rate in {{column}}: {{null_rate:.2%}}")
        
        if distinct_rate > {state.target.quality_thresholds.get('cardinality_ratio', 0.9)}:
            validation_results.append(f"High cardinality in {{column}}: {{distinct_rate:.2%}}")
    
    # Record validation results
    record_validation_results(
        layer="silver",
        validation_results=validation_results,
        quality_metrics=quality_metrics,
        timestamp=current_timestamp()
    )
    
    return validation_results, quality_metrics
"""
        steps.append(
            TransformationStep(
                name="data_validation",
                description="Validate cleaned data and generate quality metrics",
                code=validation_code,
                dependencies=["data_cleaning"],
                layer="silver",
                input_schema=self._get_cleaned_schema(state.source_schema),
                output_schema={
                    **self._get_cleaned_schema(state.source_schema),
                    "quality_metrics": "struct<column:string,metrics:struct<null_rate:double,distinct_rate:double,quality_score:double>>"
                },
                validation_rules=[
                    "Check data quality rules",
                    "Verify business constraints",
                    "Validate referential integrity",
                    "Monitor quality metrics"
                ]
            )
        )
        
        return steps

    def _generate_gold_layer(self, state: TransformerState) -> List[TransformationStep]:
        """Generate gold layer transformations for business logic."""
        steps = []
        
        if state.target.target_schema:
            # Use user-defined target schema
            steps.extend(self._generate_targeted_features(state))
        else:
            # Auto-generate comprehensive feature set
            steps.extend(self._generate_auto_features(state))
        
        return steps
    
    def _generate_targeted_features(self, state: TransformerState) -> List[TransformationStep]:
        """Generate features based on user-defined target schema."""
        steps = []
        schema = state.target.target_schema
        
        for field_name, field in schema.fields.items():
            if field.feature_config:
                # Generate specific feature as configured
                feature_code = self._generate_feature_code(field.feature_config, state.source_schema)
                steps.append(
                    TransformationStep(
                        name=f"generate_{field_name}",
                        description=f"Generate {field_name} feature",
                        code=feature_code,
                        dependencies=["data_validation"],
                        layer="gold",
                        input_schema=self._get_cleaned_schema(state.source_schema),
                        output_schema={field_name: field.type},
                        validation_rules=[
                            f"Verify {field_name} matches expected type {field.type}",
                            *field.validation_rules or []
                        ]
                    )
                )
                
                # Track feature dependencies
                state.feature_dependencies[field_name] = set(field.feature_config.source_columns)
        
        # Add final transformation to combine all features
        combine_code = self._generate_feature_combination_code(schema)
        steps.append(
            TransformationStep(
                name="combine_features",
                description="Combine all features into final schema",
                code=combine_code,
                dependencies=[step.name for step in steps],
                layer="gold",
                input_schema={step.name: step.output_schema for step in steps},
                output_schema=schema.fields,
                validation_rules=[
                    "Verify all required fields are present",
                    "Check primary key constraints",
                    "Validate field dependencies"
                ]
            )
        )
        
        return steps
    
    def _generate_auto_features(self, state: TransformerState) -> List[TransformationStep]:
        """Generate comprehensive feature set when no target schema is provided."""
        steps = []
        config = state.target.auto_feature_config
        
        # Group columns by data type
        numeric_cols = self._get_columns_by_type(state.source_schema, ['int', 'double'])
        categorical_cols = self._get_columns_by_type(state.source_schema, ['string'])
        temporal_cols = self._get_columns_by_type(state.source_schema, ['timestamp', 'date'])
        text_cols = self._identify_text_columns(state.source_schema)
        geo_cols = self._identify_geospatial_columns(state.source_schema)
        
        # Generate features by type
        if FeatureType.NUMERIC in config.enabled_feature_types:
            steps.extend(self._generate_numeric_features(numeric_cols))
        
        if FeatureType.CATEGORICAL in config.enabled_feature_types:
            steps.extend(self._generate_categorical_features(categorical_cols))
        
        if FeatureType.TEMPORAL in config.enabled_feature_types:
            steps.extend(self._generate_temporal_features(temporal_cols, config.max_window_size))
        
        if FeatureType.TEXT in config.enabled_feature_types and config.text_analysis_enabled:
            steps.extend(self._generate_text_features(text_cols))
        
        if FeatureType.GEOSPATIAL in config.enabled_feature_types and config.geospatial_analysis_enabled:
            steps.extend(self._generate_geospatial_features(geo_cols))
        
        if FeatureType.INTERACTION in config.enabled_feature_types:
            steps.extend(self._generate_interaction_features(
                numeric_cols, categorical_cols, config.max_interaction_degree
            ))
        
        # Add feature selection and correlation analysis
        steps.append(self._generate_feature_selection_step(config.correlation_threshold))
        
        return steps
    
    def _generate_numeric_features(self, columns: List[str]) -> List[TransformationStep]:
        """Generate numeric feature transformations."""
        return [
            TransformationStep(
                name="numeric_features",
                description="Generate numeric features",
                code=f"""
def generate_numeric_features(spark, df):
    # Basic statistical features
    for col in {columns}:
        window = Window.partitionBy()
        df = df.withColumn(f"{{col}}_zscore", 
            (col(col) - mean(col).over(window)) / stddev(col).over(window)
        ).withColumn(f"{{col}}_quantile",
            percent_rank().over(Window.orderBy(col))
        )
        
        # Rolling statistics if timestamp column exists
        if "timestamp" in df.columns:
            for window_size in [7, 30, 90]:
                window = Window.orderBy("timestamp").rowsBetween(-window_size, 0)
                df = df.withColumn(f"{{col}}_rolling_mean_{{window_size}}",
                    mean(col).over(window)
                ).withColumn(f"{{col}}_rolling_std_{{window_size}}",
                    stddev(col).over(window)
                )
    
    return df
""",
                layer="gold",
                dependencies=["data_validation"],
                input_schema={col: "numeric" for col in columns},
                output_schema={f"{col}_derived": "numeric" for col in columns},
                validation_rules=["Verify numeric features are not null", "Check for infinite values"]
            )
        ]
    
    def _generate_categorical_features(self, columns: List[str]) -> List[TransformationStep]:
        """Generate categorical feature transformations."""
        return [
            TransformationStep(
                name="categorical_features",
                description="Generate categorical features",
                code=f"""
def generate_categorical_features(spark, df):
    # Encoding and frequency features
    for col in {columns}:
        # Frequency encoding
        freq_df = df.groupBy(col).count()
        df = df.join(freq_df, col, "left")
        df = df.withColumn(f"{{col}}_freq", col("count") / df.count())
        
        # Target encoding if target column exists
        if "target" in df.columns and df.select("target").dtypes[0][1] in ["double", "int"]:
            target_avg = df.groupBy(col).agg(mean("target").alias("target_mean"))
            df = df.join(target_avg, col, "left")
            df = df.withColumn(f"{{col}}_target_encoded", col("target_mean"))
        
        # One-hot encoding for low cardinality
        if df.select(col).distinct().count() < 100:
            encoder = OneHotEncoder(inputCol=col, outputCol=f"{{col}}_onehot")
            df = encoder.fit(df).transform(df)
    
    return df
""",
                layer="gold",
                dependencies=["data_validation"],
                input_schema={col: "string" for col in columns},
                output_schema={f"{col}_encoded": "vector" for col in columns},
                validation_rules=["Verify encoding completeness", "Check cardinality limits"]
            )
        ]
    
    def _generate_temporal_features(self, columns: List[str], max_window: int) -> List[TransformationStep]:
        """Generate temporal feature transformations."""
        return [
            TransformationStep(
                name="temporal_features",
                description="Generate temporal features",
                code=f"""
def generate_temporal_features(spark, df):
    for col in {columns}:
        # Basic temporal features
        df = df.withColumn(f"{{col}}_year", year(col(col))) \\
            .withColumn(f"{{col}}_month", month(col(col))) \\
            .withColumn(f"{{col}}_day", dayofmonth(col(col))) \\
            .withColumn(f"{{col}}_dayofweek", dayofweek(col(col))) \\
            .withColumn(f"{{col}}_hour", hour(col(col))) \\
            .withColumn(f"{{col}}_minute", minute(col(col)))
        
        # Cyclical encoding
        df = df.withColumn(f"{{col}}_month_sin", 
            sin(col(f"{{col}}_month") * 2 * pi() / 12)
        ).withColumn(f"{{col}}_month_cos",
            cos(col(f"{{col}}_month") * 2 * pi() / 12)
        )
        
        # Time-based aggregations
        if df.select("value_column").dtypes[0][1] in ["double", "int"]:
            for window in range(1, {max_window + 1}):
                window_spec = Window.orderBy(col).rowsBetween(-window, 0)
                df = df.withColumn(f"value_{{window}}d_mean",
                    mean("value_column").over(window_spec)
                ).withColumn(f"value_{{window}}d_std",
                    stddev("value_column").over(window_spec)
                )
    
    return df
""",
                layer="gold",
                dependencies=["data_validation"],
                input_schema={col: "timestamp" for col in columns},
                output_schema={f"{col}_derived": "numeric" for col in columns},
                validation_rules=["Verify temporal feature completeness", "Check for future dates"]
            )
        ]
    
    def _generate_text_features(self, columns: List[str]) -> List[TransformationStep]:
        """Generate text feature transformations."""
        return [
            TransformationStep(
                name="text_features",
                description="Generate text features",
                code=f"""
def generate_text_features(spark, df):
    # Configure text processing
    tokenizer = Tokenizer()
    remover = StopWordsRemover()
    hashingTF = HashingTF()
    idf = IDF()
    
    for col in {columns}:
        # Basic text features
        df = df.withColumn(f"{{col}}_length", length(col(col))) \\
            .withColumn(f"{{col}}_word_count", size(split(col(col), ' '))) \\
            .withColumn(f"{{col}}_avg_word_length", 
                length(col(col)) / size(split(col(col), ' '))
            )
        
        # TF-IDF features
        tokenized = tokenizer.setInputCol(col).setOutputCol("tokens").transform(df)
        filtered = remover.setInputCol("tokens").setOutputCol("filtered").transform(tokenized)
        tf = hashingTF.setInputCol("filtered").setOutputCol("tf").transform(filtered)
        df = idf.setInputCol("tf").setOutputCol(f"{{col}}_tfidf").fit(tf).transform(tf)
        
        # Sentiment analysis if textblob is available
        try:
            from textblob import TextBlob
            sentiment_udf = udf(lambda text: TextBlob(text).sentiment.polarity, DoubleType())
            df = df.withColumn(f"{{col}}_sentiment", sentiment_udf(col(col)))
        except ImportError:
            pass
    
    return df
""",
                layer="gold",
                dependencies=["data_validation"],
                input_schema={col: "string" for col in columns},
                output_schema={f"{col}_features": "vector" for col in columns},
                validation_rules=["Verify text processing completion", "Check vector dimensions"]
            )
        ]
    
    def _generate_geospatial_features(self, columns: List[str]) -> List[TransformationStep]:
        """Generate geospatial feature transformations."""
        return [
            TransformationStep(
                name="geospatial_features",
                description="Generate geospatial features",
                code=f"""
def generate_geospatial_features(spark, df):
    # Register UDFs for geospatial calculations
    @udf(returnType=DoubleType())
    def haversine_distance(lat1, lon1, lat2, lon2):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    # Generate features for each coordinate pair
    coord_pairs = {columns}
    for lat_col, lon_col in coord_pairs:
        # Calculate distances to important locations
        for ref_lat, ref_lon, name in REFERENCE_LOCATIONS:
            df = df.withColumn(
                f"distance_to_{{name}}",
                haversine_distance(
                    col(lat_col), col(lon_col),
                    lit(ref_lat), lit(ref_lon)
                )
            )
        
        # Calculate cluster assignments
        from pyspark.ml.clustering import KMeans
        kmeans = KMeans(k=5, featuresCol=f"{{lat_col}}_{{lon_col}}_coords")
        df = kmeans.fit(df).transform(df)
    
    return df
""",
                layer="gold",
                dependencies=["data_validation"],
                input_schema={col: "double" for col in columns},
                output_schema={f"{cols[0]}_{cols[1]}_features": "vector" for cols in zip(columns[::2], columns[1::2])},
                validation_rules=["Verify coordinate validity", "Check distance calculations"]
            )
        ]
    
    def _generate_interaction_features(
        self, numeric_cols: List[str], categorical_cols: List[str], max_degree: int
    ) -> List[TransformationStep]:
        """Generate interaction feature transformations."""
        return [
            TransformationStep(
                name="interaction_features",
                description="Generate interaction features",
                code=f"""
def generate_interaction_features(spark, df):
    from itertools import combinations
    
    # Numeric interactions
    numeric_cols = {numeric_cols}
    for degree in range(2, {max_degree + 1}):
        for cols in combinations(numeric_cols, degree):
            col_name = '_x_'.join(cols)
            df = df.withColumn(
                f"{{col_name}}_interaction",
                reduce(lambda x, y: x * y, [col(c) for c in cols])
            )
    
    # Categorical interactions
    categorical_cols = {categorical_cols}
    for col1, col2 in combinations(categorical_cols, 2):
        df = df.withColumn(
            f"{{col1}}_{{col2}}_combined",
            concat_ws('_', col(col1), col(col2))
        )
    
    # Mixed interactions
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            # Calculate mean of numeric column per category
            stats = df.groupBy(cat_col).agg(
                mean(num_col).alias('mean'),
                stddev(num_col).alias('std')
            )
            df = df.join(stats, cat_col, 'left')
            
            # Calculate z-score within category
            df = df.withColumn(
                f"{{num_col}}_per_{{cat_col}}_zscore",
                (col(num_col) - col('mean')) / col('std')
            )
    
    return df
""",
                layer="gold",
                dependencies=["data_validation"],
                input_schema={
                    **{col: "numeric" for col in numeric_cols},
                    **{col: "string" for col in categorical_cols}
                },
                output_schema={
                    "interaction_features": "vector"
                },
                validation_rules=["Verify interaction completeness", "Check for multicollinearity"]
            )
        ]
    
    def _generate_feature_selection_step(self, correlation_threshold: float) -> TransformationStep:
        """Generate feature selection and correlation analysis step."""
        return TransformationStep(
            name="feature_selection",
            description="Select and validate final feature set",
            code=f"""
def select_features(spark, df):
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation
    
    # Combine all numeric columns into a feature vector
    numeric_cols = [c for c, t in df.dtypes if t in ['double', 'int']]
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    df_vector = assembler.transform(df)
    
    # Calculate correlation matrix
    correlation_matrix = Correlation.corr(df_vector, "features").collect()[0][0]
    
    # Find highly correlated features
    to_drop = set()
    for i in range(correlation_matrix.numRows):
        for j in range(i + 1, correlation_matrix.numCols):
            if abs(correlation_matrix[i, j]) > {correlation_threshold}:
                # Drop the feature with higher mean absolute correlation
                corr_i = abs(correlation_matrix[i]).mean()
                corr_j = abs(correlation_matrix[j]).mean()
                to_drop.add(numeric_cols[i if corr_i > corr_j else j])
    
    # Drop highly correlated features
    final_cols = [c for c in numeric_cols if c not in to_drop]
    
    # Calculate feature importance if target column exists
    if "target" in df.columns:
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.regression import RandomForestRegressor
        
        assembler = VectorAssembler(inputCols=final_cols, outputCol="features")
        rf = RandomForestRegressor(featuresCol="features", labelCol="target")
        model = rf.fit(assembler.transform(df))
        
        # Sort features by importance
        feature_importance = sorted(
            zip(final_cols, model.featureImportances),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Keep top features
        final_cols = [f[0] for f in feature_importance[:100]]  # Adjust limit as needed
    
    return df.select(*final_cols)
""",
            layer="gold",
            dependencies=["numeric_features", "categorical_features", "temporal_features",
                         "text_features", "geospatial_features", "interaction_features"],
            input_schema={"features": "vector"},
            output_schema={"selected_features": "vector"},
            validation_rules=[
                "Verify feature independence",
                "Check feature importance scores",
                "Validate final feature count"
            ]
        )
    
    def _get_columns_by_type(self, schema: Dict[str, Any], types: List[str]) -> List[str]:
        """Get columns of specified types from schema."""
        return [
            col for col, info in schema.items()
            if any(t in str(info['type']).lower() for t in types)
        ]
    
    def _identify_text_columns(self, schema: Dict[str, Any]) -> List[str]:
        """Identify columns containing text data."""
        return [
            col for col, info in schema.items()
            if info['type'] == 'string' and
            info.get('avg_length', 0) > 50  # Adjust threshold as needed
        ]
    
    def _identify_geospatial_columns(self, schema: Dict[str, Any]) -> List[str]:
        """Identify columns containing geospatial data."""
        geo_patterns = ['lat', 'lon', 'latitude', 'longitude', 'coord']
        return [
            col for col in schema
            if any(pattern in col.lower() for pattern in geo_patterns)
        ] 