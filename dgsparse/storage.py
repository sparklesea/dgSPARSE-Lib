from typing import Optional

import torch


class Storage(object):
    _row: Optional[torch.Tensor]
    _rowptr: Optional[torch.Tensor]
    _col: Optional[torch.Tensor]
    _values: Optional[torch.Tensor]
    _colptr: torch.Tensor
    _csr2csc: torch.Tensor
    _csc2csr: torch.Tensor
    _colcount: Optional[torch.Tensor]
    _group_rowptr: Optional[torch.Tensor]
    _group_row: Optional[torch.Tensor]
    _group_colptr: Optional[torch.Tensor]
    _group_col: Optional[torch.Tensor]

    def __init__(
        self,
        row: Optional[torch.Tensor] = None,
        rowptr: Optional[torch.Tensor] = None,
        col: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        colptr: Optional[torch.Tensor] = None,
        csr2csc: Optional[torch.Tensor] = None,
        csc2csr: Optional[torch.Tensor] = None,
        colcount: Optional[torch.Tensor] = None,
        group_rowptr: Optional[torch.Tensor] = None,
        group_row: Optional[torch.Tensor] = None,
        group_colptr: Optional[torch.Tensor] = None,
        group_col: Optional[torch.Tensor] = None,
    ):
        assert row is not None or rowptr is not None
        assert col is not None
        assert col.dtype == torch.int
        assert col.dim() == 1
        col = col.contiguous()

        M: int = 0
        if rowptr is not None:
            M = rowptr.numel() - 1
        elif row is not None and row.numel() > 0:
            M = int(row.max()) + 1

        N: int = 0
        if col.numel() > 0:
            N = int(col.max()) + 1

        self.sparse_sizes = (M, N)
        self.nnz = col.size(0)

        if row is not None:
            assert row.dtype == torch.int
            assert row.device == col.device
            assert row.dim() == 1
            assert row.numel() == col.numel()
            row = row.contiguous()

        if rowptr is not None:
            assert rowptr.dtype == torch.int
            assert rowptr.device == col.device
            assert rowptr.dim() == 1
            assert rowptr.numel() - 1 == self.sparse_sizes[0]
            rowptr = rowptr.contiguous()

        if values is not None:
            assert values.device == col.device
            assert values.size(0) == self.nnz
            values = values.contiguous()
        else:
            values = torch.ones((self.nnz),
                                dtype=torch.float,
                                device=col.device)
            values = values.contiguous()

        if colptr is not None:
            assert colptr.dtype == torch.long
            assert colptr.device == col.device
            assert colptr.dim() == 1
            assert colptr.numel() - 1 == self.sparse_sizes[1]
            colptr = colptr.contiguous()

        if csr2csc is not None:
            assert csr2csc.dtype == torch.long
            assert csr2csc.device == col.device
            assert csr2csc.dim() == 1
            assert csr2csc.numel() == col.size(0)
            csr2csc = csr2csc.contiguous()

        if colcount is not None:
            assert colcount.dtype == torch.long
            assert colcount.device == col.device
            assert colcount.dim() == 1
            assert colcount.numel() == self.sparse_sizes[1]
            colcount = colcount.contiguous()

        self._row = row
        self._rowptr = rowptr
        self._col = col
        self._values = values
        self._colptr = colptr
        self._csr2csc = csr2csc
        self._colcount = colcount
        self._group_rowptr = group_rowptr
        self._group_row = group_row
        self._group_colptr = group_colptr
        self._group_col = group_col
        # convert
        self.csr2csc_convert()

    @classmethod
    def empty(self):
        row = torch.tensor([], dtype=torch.int)
        col = torch.tensor([], dtype=torch.int)
        return Storage(
            row=row,
            rowptr=None,
            col=col,
            values=None,
            colptr=None,
            csc2csr=None,
            csr2csc=None,
            colcount=None,
        )

    def row(self) -> torch.Tensor:
        row = self._row
        if row is not None:
            return row
        else:
            raise ValueError

    def rowptr(self) -> torch.Tensor:
        rowptr = self._rowptr
        if rowptr is not None:
            return rowptr
        else:
            raise ValueError

    def col(self) -> torch.Tensor:
        col = self._col
        if col is not None:
            return col
        else:
            raise ValueError

    def colptr(self) -> torch.Tensor:
        colptr = self._colptr
        if colptr is not None:
            return colptr
        else:
            raise ValueError

    def values(self) -> torch.Tensor:
        values = self._values
        if values is not None:
            return values
        else:
            raise ValueError

    def csr2csc(self) -> torch.Tensor:
        csr2csc = self._csr2csc
        if csr2csc is not None:
            return csr2csc
        else:
            raise ValueError

    def csr2csc_convert(self):
        if self._csr2csc is not None:
            return self._csr2csc
        device = self._col.device
        # idx = torch.range(0, 100, device=device)
        idx = torch.arange(self._col.shape[0],
                           device=device,
                           dtype=torch.float)
        colptr, row, csr2csc = torch.ops.dgsparse_spmm.csr2csc(
            self._rowptr, self._col, idx)
        csr2csc = csr2csc.to(torch.int)
        if self._row is None:
            self._row = row
        if self._colptr is None:
            self._colptr = colptr
        self._csr2csc = csr2csc

    def to_group_tensor(self, group_size):
        assert self._rowptr is not None
        # if self._group_rowptr is None or self._group_row is None
        # or self._group_colptr is None or self._group_col is None:
        if self._colptr is None:
            colptr, _, _ = torch.ops.dgsparse_spmm.csr2csc(
                self._rowptr, self._col, self._values)
            self._colptr = colptr

        group_rowptr = []
        group_row = []

        for rid in range(self._rowptr.numel() - 1):
            high = self._rowptr[rid + 1]
            tmp_rowptr = self._rowptr[rid].item()
            while tmp_rowptr < high:
                group_rowptr.append(tmp_rowptr)
                tmp_rowptr += group_size
                group_row.append(rid)
        if group_rowptr[-1] != self._rowptr[-1]:
            group_rowptr.append(self._rowptr[-1].item())

        group_colptr = []
        group_col = []
        for rid in range(self._colptr.numel() - 1):
            high = self._colptr[rid + 1]
            tmp_colptr = self._colptr[rid].item()
            while tmp_colptr < high:
                group_colptr.append(tmp_colptr)
                tmp_colptr += group_size
                group_col.append(rid)
        if group_colptr[-1] != self._colptr[-1]:
            group_colptr.append(self._colptr[-1].item())

        self._group_rowptr = torch.tensor(group_rowptr,
                                          dtype=torch.int,
                                          device=self._col.device,
                                          requires_grad=False)
        self._group_row = torch.tensor(group_row,
                                       dtype=torch.int,
                                       device=self._col.device,
                                       requires_grad=False)

        self._group_colptr = torch.tensor(group_colptr,
                                          dtype=torch.int,
                                          device=self._col.device,
                                          requires_grad=False)
        self._group_col = torch.tensor(group_col,
                                       dtype=torch.int,
                                       device=self._col.device,
                                       requires_grad=False)
