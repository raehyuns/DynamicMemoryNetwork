# DynamicMemoryNetwork

# Single Supporting Fact
## Train
<p align="center">
  <img src=https://github.com/Kong26/DynamicMemoryNetwork/blob/master/Results/Single_Supporting_Fact_Train_history.PNG width="1200"/>
</p>

## Test
<p align="center">
  <img src=https://github.com/Kong26/DynamicMemoryNetwork/blob/master/Results/Single_Supporting_Fact_Test_result.PNG width="1200"/>
</p>

# Qa4 Two Arg Relations
## Train
<p align="center">
  <img src=https://github.com/Kong26/DynamicMemoryNetwork/blob/master/Results/qa4_training.PNG width="1200"/>
</p>

## Test
<p align="center">
  <img src=https://github.com/Kong26/DynamicMemoryNetwork/blob/master/Results/qa4_test.PNG width="1200"/>
</p>



### GRU input module    
torch.Size([3000, 300])  
torch.Size([3000, 1000])  
torch.Size([3000])  
torch.Size([3000])  
    
### GRU question module
torch.Size([3000, 300])  
torch.Size([3000, 1000])  
torch.Size([3000])  
torch.Size([3000])  

### attention zW  
torch.Size([1000, 1000])

### attention gW
torch.Size([1000, 7002])  
torch.Size([1000])  
torch.Size([1, 1000])  
torch.Size([1])  
  
### GRUcell_e    
torch.Size([3000, 1000])  
torch.Size([3000, 1000])  
torch.Size([3000])  
torch.Size([3000])  
  
### GRUcell_m    
torch.Size([3000, 1000])  
torch.Size([3000, 1000])  
torch.Size([3000])  
torch.Size([3000])  

### Wa_Linear    
torch.Size([159, 1000])  

### GRUcell_a    
torch.Size([3000, 1159])  
torch.Size([3000, 1000])  
torch.Size([3000])  
torch.Size([3000])  

