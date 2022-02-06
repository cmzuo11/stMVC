plot_colors=c("1" = "#6D1A9C", "2" = "#CC79A7","3"  = "#7495D3", "4" = "#59BE86", "5" = "#56B4E9", "6" = "#FEB915", 
              "7" = "#DB4C6C", "8" = "#C798EE", "9" = "#3A84E6", "10"= "#FF0099FF", "11" = "#CCFF00FF",
              "12" = "#268785", "13"= "#FF9900FF", "14"= "#33FF00FF", "15"= "#AF5F3C", "16"= "#DAB370", 
              "17" = "#554236", "18"= "#787878", "19"= "#877F6C")

Seurat_processing = function(basePath, robust_rep, nDim = 10, nCluster = 7, save_path = NULL, pdf_file = NULL ){
  
  library("Seurat")
  library('ggplot2')
  
  idc = Load10X_Spatial(data.dir= basePath )
  idc = SCTransform(idc, assay = "Spatial", verbose = FALSE)
  idc = RunPCA(idc, assay = "SCT", verbose = FALSE)
  
  input_features = as.matrix(robust_rep[match(colnames(idc), row.names(robust_rep)),])
  original_pca_emb           = idc@reductions$pca@cell.embeddings
  row.names(input_features)  = row.names(original_pca_emb)
  idc@reductions$pca@cell.embeddings[,1:nDim] = input_features
  
  idc = FindNeighbors(idc, reduction = "pca", dims = 1:nDim)
  
  for(qq in seq(0.05,1.5,0.01))
  {
    idc <- FindClusters( idc, resolution = qq,  verbose = FALSE )
    if(length(table(Idents(idc)))==nCluster)
    {
      break
    }
  }
  
  idc = RunUMAP(idc, reduction = "pca", dims = 1:nDim)
  idc[["clusterings"]] = as.character(as.numeric(as.character(Idents(idc)))+1)
  
  pdf( paste0( save_path, pdf_file ), width = 10, height = 10)
  
  p1  = DimPlot(idc, reduction = "umap", label = T, label.size = 6, pt.size=1.5,
                cols = plot_colors, group.by = "clusterings")+
    theme(legend.position = "none",
          legend.title = element_blank())+
    ggtitle("")
  print(p1)
  
  p2  = SpatialDimPlot(idc, label = T, label.size = 3, cols = plot_colors, 
                       group.by = "clusterings" )+
    theme(legend.position = "none",
          legend.title = element_blank())+
    ggtitle("")
  
  print(p2)
  dev.off()
  
  return(idc)
}

# K-nearest neighbor smoothing for high-throughput RNA-Seq data
# Authos: Chunman Zuo, changed the script from <yun.yan@nyumc.org>
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(rsvd))

pdist = function(tmat){
  # @param tmat A non-negative matrix with samples by features
  # @reference http://r.789695.n4.nabble.com/dist-function-in-R-is-very-slow-td4738317.html
  mtm            = Matrix::tcrossprod(tmat)
  sq             = rowSums(tmat^2)
  out0           = outer(sq, sq, "+") - 2 * mtm
  out0[out0 < 0] = 0
  sqrt(out0)
}

smoother_aggregate_nearest_nb = function(mat, D, k){
  sapply(seq_len(ncol(mat)), function(cid){
    nb_cid      = head(order(D[cid, ]), k)
    closest_mat = mat[, nb_cid, drop=FALSE]
    return(Matrix::rowSums(closest_mat))
  })
}

knn_smoothing = function(mat, k = 15, latent_matrix, seed=42){
  # KNN-smoothing on UMI-filtered single-cell RNA-seq data
  S         = mat
  D         = pdist(latent_matrix)
  S         = smoother_aggregate_nearest_nb(mat, D, k)
  colnames(S) = colnames(mat)
  rownames(S) = rownames(mat)
  
  return(S)
}

Annotation_random_split = function(basePath, anno_file, anno_file_split, prop_seq = 0.3){
  
  data             = read.csv(paste0(basePath, anno_file), header=T, row.names=1)
  unique_cluster   = unique( as.character(data[[1]]) )
  
  test_cells       = train_cells = NULL
  cell_clusrer_nos = rep(0, dim(data)[1])
  for(i in 1:length(unique_cluster))
  {
    cells_uni_clu = row.names(data)[which(  as.character(data[[1]]) == unique_cluster[i])]
    test_cells_te = sample( cells_uni_clu, floor(length(cells_uni_clu)*0.3) )
    test_cells    = c(test_cells, test_cells_te)
    train_cells   = c(train_cells, setdiff(cells_uni_clu, test_cells_te))
    cell_clusrer_nos[which( as.character(data[[1]]) == unique_cluster[i])] = i
  }
  data[[2]]         = cell_clusrer_nos
  train_trest_state = rep(0, dim(data)[1])
  train_trest_state[match(train_cells, row.names(data))] = 1
  data[[3]]         = train_trest_state
  names(data)       = c("Layer", "Cluster", "Train_test")
  write.csv(data, file = paste(work, "Image_segmentation_cell_label_all_0.5_train_test_split_0.3.csv", sep=""), quote=F)
  
  test_cells     = train_cells = NULL
  for(i in 1:length(unique_cluster))
  {
    cells_uni_clu = row.names(data)[which(  as.character(data[[1]]) == unique_cluster[i])]
    test_cells_te = sample( cells_uni_clu, floor(length(cells_uni_clu)*prop_seq) )
    test_cells    = c(test_cells, test_cells_te)
    train_cells   = c(train_cells, setdiff(cells_uni_clu, test_cells_te))
  }
  train_trest_state = rep(0, dim(data)[1])
  train_trest_state[match(train_cells, row.names(data))] = 1
  data[[4]]         = train_trest_state
  
  names(data) = c("Layer", "Cluster", "Train_test", "Train_test_prop")
  write.csv(data, file = paste(basePath, anno_file_split, sep=""), quote=F)
}


#Further analysis
basePath       = "./stMVC_test_data/DLPFC_151673/"
robust_rep     = read.csv( paste0(basePath, "stMVC/GAT_2-view_robust_representation.csv"), header = T, row.names = 1)
Seurat_obj     = Seurat_processing(basePath, robust_rep, 10, 7, basePath, "stMVC/stMVC_clustering.pdf" )

input_features = as.matrix(robust_rep[match(colnames(Seurat_obj), row.names(robust_rep)),])
Seurat_obj     = FindVariableFeatures(Seurat_obj, nfeatures=2000)
hvg            = VariableFeatures(Seurat_obj)
rna_data       = as.matrix(Seurat_obj@assays$Spatial@counts)
hvg_data       = rna_data[match(hvg, row.names(rna_data)), ]

mat_smooth     = knn_smoothing( hvg_data, 15, input_features )
colnames(mat_smooth) = colnames(Seurat_obj)

Seurat_smooth         = CreateSeuratObject(counts=mat_smooth, assay='Spatial')
Idents(Seurat_smooth) = Idents(Seurat_obj)

Seurat_smooth = SCTransform(Seurat_smooth, assay = "Spatial", verbose = FALSE)
top_markers   = FindAllMarkers(Seurat_smooth, assay='SCT', slot='data',only.pos=TRUE) 
Seurat_smooth[["clusterings"]] = as.numeric(as.character(Idents(Seurat_smooth)))+1


